/*
 *   Copyright (c) 2026 Australian National University
 *   All rights reserved.
 */

#include "./baldr.h"
#include "commander/commander.h"
#include "baldr.h"
//#define PRINT_TIMING

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

    if (sz != SUBARRAY_WIDTH)
    {
        throw std::runtime_error(
            "Subarray is not same width as constant SUBARRAY_WIDTH; " +
            std::to_string(sz) + " != " +
            std::to_string(SUBARRAY_WIDTH) + "\n");
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
    catch_up_with_sem(&subarray, 1);

    // infinite loop while servo is running (not necessarily closed loop)
    int servo_mode, last_servo_mode = SERVO_OFF;
    while (true)
    {
        last_servo_mode = servo_mode;
        settings.mutex.lock();
        servo_mode = settings.settings.servo_mode;
        settings.mutex.unlock();
        cnt_since_init++; // This should "never" wrap around, as a long int is big.
        // See if there was a semaphore signalled for the next frame to be ready in K1 and K2
        ImageStreamIO_semwait(&subarray, 1);
#ifdef PRINT_TIMING
        auto t1 = high_resolution_clock::now();
#endif
        // Image is ready, read it from shm
        read_shm();
        // Compute some monitoring variables for the supervisor
        rt_status.mutex.lock();
        ctrl.mutex.lock();
        rt_status.status.flux = ctrl.flux_est;
        ctrl.mutex.unlock();
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

        // apply the reconstructor to estimate the mode values from the
        // calibrated measurement
        reconstruct_modes();

        // filter/integrate the reconstructed modes to produce a good clean
        // compensatory set of modes.
        filter_modes(servo_mode);

        // project the modes into the command space
        project_com();

        // clip commands to ensure correct internal loop feedback and reduce
        // DM uncertainties
        clip_com();

        // inject a disturbance (nominally just zeros)
        inject_disturb(servo_mode);

        // write to shared memory and post the semaphore for that DM shmim
        if (servo_mode != SERVO_OFF || last_servo_mode != SERVO_OFF) 
        {
            write_shm();
        } 
#ifdef PRINT_TIMING
        auto t2 = high_resolution_clock::now();
        if (cnt % 20 == 0)
        {
            info("|----------|------------|");
            info("|    cnt   |  critical  |");
            info("|----------|------------|");
        }
        duration<double, std::nano> ns_double = (t2 - t1);
        std::string msg = fmt::format("| {:8} | {:6} ns |", cnt, ns_double.count());
        info(msg.c_str());
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
    // perform strehl and flux estimation
    // note: this would likely be more optimised if we convert the shmim to a
    // Eigen3 matrix type, then perform the two operations as dot products, but
    // until I can identify that this is a bottleneck then I'll keep it simple.
    ctrl.flux_est = 0.0;
    ctrl.strehl_est = 0.0;
    for (size_t i = 0; i < N_SUBARRAY_PIXELS; i++)
    {
        double element = (double)subarray.array.SI32[i];
        ctrl.flux_est += ctrl.flux_mask(i, 0) * element;
        ctrl.strehl_est += ctrl.strehl_mask(i, 0) * element;
    }
    if (ctrl.flux_est <= 0.0)
    {
        throw std::runtime_error("flux estimate is equal to zero, quitting now to avoid div by 0");
    }
    ctrl.strehl_est /= ctrl.flux_est;
    ctrl.cnt = cnt;
    ctrl.mutex.unlock();
}

void calibrate_frame()
{
    ctrl.mutex.lock();
    // First, we divide the full frame by the flux estimate
    ctrl.meas_norm = ctrl.meas_raw / ctrl.flux_est;

    // eventually, the meas_offset needs to be computed based on a strehl-indexed
    // lookup table. Until then, we have a static measurement offset computed
    // during interaction matrix computation; and we just print the strehl
    // estimate out.
    // info("| %lu | flux = %5.2e | sre = %5.2e |", ctrl.cnt, ctrl.flux_est, ctrl.strehl_est);
    // the closed-loop calibrated measurement is the normalised measurement plus
    // the measurement offset (typically the negative of the reference
    // measurement, but may also be a function of NCPAs).
    ctrl.meas_cl = ctrl.meas_norm + ctrl.meas_offset;
    ctrl.mutex.unlock();
}

void reconstruct_modes()
{
    ctrl.mutex.lock();
    // the reconstructed modes are the matrix-vector product of the
    // reconstructor matrix (meas_to_modes) and the pseudo-open loop
    // measurements
    ctrl.mode_raw = ctrl.meas_to_mode * ctrl.meas_cl;
    ctrl.mutex.unlock();
}

void filter_modes(int servo_mode)
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
        ctrl.mode_raw_buffer.row(i).swap(ctrl.mode_raw_buffer.row(i - 1));
    }
    ctrl.mode_raw_buffer.row(0) = ctrl.mode_raw;

    // INITIALLY ZERO THE OUTPUT COMING FROM THIS CALCULATION
    ctrl.mode_filt.setZero();

    // This is an unconventional way to open the loop, but I'd like to try it.
    // The logic is that measurements propagate all the way to the IIR filter
    // always, but the filter is bypassed if the loop is "open". The IIR output
    // buffer still updates with zeroes in that case.
    // This design choice allows telemetry to flow as normal, until the IIR
    // and is equivalent to setting the IIR coefficients to zeros, except that
    // the IIR coffecients don't need to be modified.
    if (servo_mode == SERVO_CLOSED)
    {
        // COMPUTE COMPONENT FROM INPUTS
        // add the input part of the IIR filter to the current output
        // NOTE: This can be done by matrix multiplication, this is just a first
        // pass to get the pipeline sound.
        for (size_t i = 0; i < FILTER_LEN; i++)
        {
            ctrl.mode_filt += (ctrl.mode_raw_buffer.row(i).array() * ctrl.filter_coeff_in.row(i).array()).matrix();
        }

        // COMPUTE COMPONENT FROM OUTPUTS
        // same for outputs, note there is one less coefficient on the output filter
        for (size_t i = 0; i < FILTER_LEN; i++)
        {
            ctrl.mode_filt += (ctrl.mode_filt_buffer.row(i).array() * ctrl.filter_coeff_out.row(i).array()).matrix();
        }

        // apply anti-windup saturations:
        ctrl.mode_filt = (ctrl.mode_filt.array().min(ctrl.mode_max).max(ctrl.mode_min)).matrix();
    }

    
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

void inject_disturb(int servo_mode)
{
    ctrl.mutex.lock();
// add the next disturbance buffer element to the command vector
#if DIST_LEN > 0
    ctrl.com_write = ctrl.com_clean + ctrl.com_dist_buffer.col(cnt % DIST_LEN);
#else
    ctrl.com_write = ctrl.com_clean;
#endif
    if (servo_mode == SERVO_OFF)
        ctrl.com_write.setZero();
    ctrl.mutex.unlock();
}

void write_shm()
{
    ctrl.mutex.lock();
    // write to the dm shm
    for (size_t i = 0; i < N_ACTUATORS; i++)
    {
        // TODO: Do we also need to post to this shmim semaphore, or is it
        // sufficient to do only the master DM? On reading the DM server code
        // for the high performance server, only the master DM semaphore matters.
        DM_high.array.D[i] = ctrl.com_write[i];
    }
    ctrl.mutex.unlock();

    // Where is the semaphore index defined?
    // Poke the master DM to trigger an update.
    ImageStreamIO_sempost(&master_DM, 1);
}
