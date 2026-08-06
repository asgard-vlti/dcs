/*
 *   Copyright (c) 2026 Australian National University
 *   All rights reserved.
 */

#include "./baldr.h"
#include "commander/commander.h"
#include "baldr.h"
#define PRINT_TIMING

#ifdef PRINT_TIMING
#include <chrono>
#endif
// #define PRINT_TIMING_ALL
// #define DEBUG
// #define DEBUG_FILTER6

uint64_t cnt = 0, cnt_since_init = 0;
int64_t nerrors = 0;
int64_t low_flux = 0;
size_t sz = 0;
double *window, *subim;
std::mutex im_mutex;

// Initialise variables and arrays on startup
void initialise_servo()
{
    cnt_since_init = 0;
    // Check the subarray.
    if (subarray.md->naxis != 2)
    {
        throw std::runtime_error("Subarray is not 2D");
    }
    sz = subarray.md->size[0];
    if (subarray.md->size[1] != sz)
    {
        throw std::runtime_error("Subarray is not square");
    }

    // Initialise the control variables
    reset_ctrl();
    write_shm();
    ImageStreamIO_sempost(&master_DM, 1);
}

//------------------------------------------------------------------------------
// Drain any outstanding semaphore posts so that
// the next semwait() really waits for a fresh frame.
//------------------------------------------------------------------------------
static inline void catch_up_with_sem(IMAGE *img, int semid)
{
    // keep grabbing until there are no more pending posts
    while (ImageStreamIO_semtrywait(img, semid) == 0)
    { /* nothing just do it*/
        ;
    }
}

// The main AO servo loop
void servo_loop()
{
#ifdef PRINT_TIMING
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::microseconds;
#endif
    // initialise servo loop
    initialise_servo();

    // global cnt variable initialised to subarray cnt0
    cnt = subarray.md->cnt0;

    // TODO: why semid 2, where is that defined?
    catch_up_with_sem(&subarray, 2);

    // infinite loop while servo is running (not necessarily closed loop)
    while (settings.settings.servo_mode != SERVO_STOP)
    {
        cnt_since_init++; // This should "never" wrap around, as a long int is big.

        // See if there was a semaphore signalled for the next frame to be ready in K1 and K2
        ImageStreamIO_semwait(&subarray, 2);
#ifdef PRINT_TIMING
        auto t1 = high_resolution_clock::now();
#endif
        // Image is ready, read it from shm
        read_shm();

        // Compute some monitoring variables for the supervisor
        rt_status.mutex.lock();
        rt_status.status.flux = ctrl.meas_raw.sum();
        rt_status.status.nerrors = nerrors;
        if (rt_status.status.flux < settings.settings.flux_threshold)
        {
            rt_status.status.nlowflux++;
            rt_status.mutex.unlock();
            continue;
        }
        else
        {
            rt_status.mutex.unlock();
        }

        // If the flux is above the threshold, run an interation of the
        // controller and update the DM image.

        // Remove reference image
        calibrate_frame();

        // infer the pseudo-open-loop measure
        compute_pol_meas();

        // apply the reconstructor to estimate the mode values from the
        // calibrated pseudo-open loop measurement
        reconstruct_modes();

        // filter the reconstructed modes to produce a good clean compensatory
        // set of modes.
        filter_modes();

        // project the modes into the command space
        project_com();

        // clip commands to ensure correct internal loop feedback and reduce
        // DM uncertainties
        clip_com();

        // inject a disturbance (nominally just zeros)
        inject_disturb();

        // write to shared memory and post the semaphore for that DM shmim
        write_shm();
#ifdef PRINT_TIMING
        auto t2 = high_resolution_clock::now();
#endif
        remove_offset();

        inject_dm_signal();

#ifdef PRINT_TIMING
        if (cnt % 20 == 0) {
            std::cout << "|----------|-----------|-----------|\n";
            std::cout << "|    cnt   |  critical |   total   |\n";
            std::cout << "|----------|-----------|-----------|\n";
        }
        auto t3 = high_resolution_clock::now();
        duration<double, std::micro> us_double = (t2 - t1);
        printf("| %8d | %6.1f us", cnt, us_double.count());
        us_double = (t3 - t1);
        printf(" | %6.1f us |\n", us_double.count());
#endif
    }
}

void read_shm()
{
    // If we are here, then a new frame is available in both K1 and K2.
    // Check that there has not been a counting error.
    if (subarray.md->cnt0 == cnt)
    {
        info("FT: Semaphore signalled but no new frame");
        nerrors++;
    }
    // Check for missed frames
    // TODO: shouldnt this be >= ? otherwise we are assuming 2 missed frames
    if (subarray.md->cnt0 > cnt + 2)
    {
        info("Missed frames! Image: %llu Servo: %lu", (unsigned long long)subarray.md->cnt0, cnt);
        // Catch up!
        catch_up_with_sem(&subarray, 2);
        cnt = subarray.md->cnt0 - 1;
        nerrors++;
    }
    cnt++;

    ctrl.mutex.lock();
    // Copy the data from the IMAGE subarray to the subimage.
    for (size_t ii = 0; ii < WIDTH; ii++)
    {
        for (size_t jj = 0; jj < WIDTH; jj++)
        {
            int y = settings.settings.py - WIDTH / 2 + ii;
            int x = settings.settings.px - WIDTH / 2 + jj;
            ctrl.meas_raw(ii * WIDTH + jj) = (double)(subarray.array.SI32[y * sz + x]);
        }
    }
    ctrl.cnt = cnt;
    ctrl.mutex.unlock();
}

void calibrate_frame()
{
    ctrl.mutex.lock();
    // the closed-loop calibrated measurement is the raw measurement plus
    // the measurement offset (typically the negative of the reference
    // measurement, but may also be a function of NCPAs).
    ctrl.meas_cl = ctrl.meas_raw + ctrl.meas_offset;
    ctrl.mutex.unlock();
}

void compute_pol_meas()
{
    ctrl.mutex.lock();
    ctrl.meas_pol = ctrl.meas_cl + ctrl.meas_feedback;
    ctrl.mutex.unlock();
}

void reconstruct_modes()
{
    ctrl.mutex.lock();
    // the reconstructed modes are the matrix-vector product of the
    // reconstructor matrix (meas_to_modes) and the pseudo-open loop
    // measurements
    ctrl.mode_pol = ctrl.meas_to_mode * ctrl.meas_pol;
    ctrl.mutex.unlock();
}

void filter_modes()
{
    ctrl.mutex.lock();

    // IIR filter:
    // We define the 0th column of the mode_filt_buffer to be the
    // current output of the IIR filter to be applied this iteration
    // to the DM.

    // CYCLE INPUT BUFFER
    // compose the mode_pol_buffer by shuffling the existing components
    // and setting the zeroth component to be the current mode_pol
    for (size_t i = FILTER_LEN - 1; i > 0; i--)
    {
        ctrl.mode_pol_buffer.row(i).swap(ctrl.mode_pol_buffer.row(i - 1));
    }
    ctrl.mode_pol_buffer.row(0) = ctrl.mode_pol;

    // INITIALLY ZERO THE OUTPUT COMING FROM THIS CALCULATION
    ctrl.mode_filt.setZero();

    // COMPUTE COMPONENT FROM INPUTS
    // add the input part of the IIR filter to the current output
    // NOTE: This can be done by matrix multiplication, this is just a first
    // pass to get the pipeline sound.
    for (size_t i = 0; i < FILTER_LEN; i++)
    {
        ctrl.mode_filt += (ctrl.mode_pol_buffer.row(i).array() * ctrl.filter_coeff_in.row(i).array()).matrix();
    }

    // COMPUTE COMPONENT FROM OUTPUTS
    // same for outputs, note there is one less coefficient on the output filter
    for (size_t i = 0; i < FILTER_LEN; i++)
    {
        ctrl.mode_filt += (ctrl.mode_filt_buffer.row(i).array() * ctrl.filter_coeff_out.row(i).array()).matrix();
    }

    // apply anti-windup saturations:
    ctrl.mode_filt = (ctrl.mode_filt.array().min(ctrl.mode_max).max(ctrl.mode_min)).matrix();

    // apply modal offset:
    ctrl.mode_filt = ctrl.mode_filt + ctrl.mode_offset;

    // CYCLE OUTPUT BUFFER
    // shuffle the filter buffer and set the current output to zero.
    for (size_t i = FILTER_LEN - 1; i > 0; i--)
    {
        ctrl.mode_filt_buffer.row(i).swap(ctrl.mode_filt_buffer.row(i - 1));
    }
    ctrl.mode_filt_buffer.row(0) = ctrl.mode_filt;

    ctrl.mutex.unlock();
}

void project_com()
{
    ctrl.mutex.lock();
    // project the filtered modes to the command space
    ctrl.com_raw = ctrl.mode_to_com * ctrl.mode_filt;
    ctrl.mutex.unlock();
}

void clip_com()
{
    ctrl.mutex.lock();
    ctrl.com_clean = ctrl.com_raw.array().min(ctrl.com_max).max(ctrl.com_min).matrix();
    ctrl.mutex.unlock();
}

void inject_disturb()
{
    ctrl.mutex.lock();
    // add the next disturbance buffer element to the command vector
    ctrl.com_write = ctrl.com_clean + ctrl.com_dist_buffer.col(cnt % DIST_LEN);
    ctrl.mutex.unlock();
}

void write_shm()
{
    ctrl.mutex.lock();
    // write to the dm shm
    for (size_t i = 0; i < N_ACTUATORS; i++)
    {
        // TODO: Do we also need to post to this shmim semaphore, or is it
        // sufficient to do only the master DM?
        DM_low.array.D[i] = ctrl.com_write[i];
    }
    ctrl.mutex.unlock();

    // Where is the semaphore index defined?
    // Poke the master DM to trigger an update.
    ImageStreamIO_sempost(&master_DM, 1);
}

void remove_offset()
{
    ctrl.mutex.lock();
    ctrl.com_feedback = ctrl.com_clean + ctrl.com_offset;
    ctrl.mutex.unlock();
}

void inject_dm_signal()
{
    // this task computes the effective DM signal on the measurement that
    // will be received next frame.

    // ctrl.delay is some real number between 0 and (say) 10
    // A value of 0 implies that there is no delay - i.e., that the
    // command we have just sent will be seen in the measurement we are
    // about to receive. This is of course not practical, since
    // the WFS takes a full frame to integrate and then another ~frame to
    // readout. The DM also takes time to move, and then there are network
    // delays. We will measure the system delay and save it in "ctrl.delay".
    //
    // In general, we want to assume that the delay can be fractional, e.g.,
    // delay=2.5 implies that the 3rd and 4th most recent commands will have
    // influence on the measurement we are about to receive. We will linearly interpolate
    // those commands in order to get something close to the true shape of the
    // DM during the exposure.
    //
    // For a delay of 2.5, we require a buffer of the:
    //   - most recent command, (to compute the 2nd most recent command next frame)
    //   - 2nd most recent command, (to compute the 3rd most recent command next frame)
    //   - 3rd most recent command, (for the delay calc, and to compute the 4th mrcnf)
    //   - and 4th most recent command. (for the delay calc (can be dropped afterwards)).
    //
    // The buffer is defined as having the most recent command as the 0th column, and so on.
    ctrl.mutex.lock();
    for (size_t i = COM_BUFFER_LEN - 1; i > 0; i--)
    {
        ctrl.com_fb_buffer.col(i).swap(ctrl.com_fb_buffer.col(i - 1));
    }
    ctrl.com_fb_buffer.col(0) = ctrl.com_feedback.col(0);
    int idx_a = floor(ctrl.delay);
    double remainder = ctrl.delay - (double)idx_a;
    ctrl.com_effective = ctrl.com_fb_buffer.col(idx_a) * (1 - remainder) + ctrl.com_fb_buffer.col(idx_a + 1) * (remainder);
    ctrl.meas_feedback = -ctrl.com_to_meas * ctrl.com_effective;
    ctrl.mutex.unlock();
}
