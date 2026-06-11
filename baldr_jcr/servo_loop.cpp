/*
 *   Copyright (c) 2026 Australian National University
 *   All rights reserved.
 */

#include "./baldr.h"
#include "commander/commander.h"
#include "baldr.h"
// #define PRINT_TIMING
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
    shm_write();
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
        } else {
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

        // inject a disturbance (nominally just zeros)
        inject_disturb();

        // write to shared memory and post the semaphore to trigger the DM
        // controller.
        shm_write();
        ImageStreamIO_sempost(&master_DM, 1);
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
    ctrl.mutex.unlock();
}

void calibrate_frame()
{
    ctrl.mutex.lock();
    // the closed-loop calibrated measurement is the raw measurement minus
    // the reference frame
    ctrl.meas_cl = ctrl.meas_raw - ctrl.meas_ref;
    ctrl.mutex.unlock();
}

void compute_pol_meas()
{
    ctrl.mutex.lock();
    // TODO this one requires some finessing with the delay
    ctrl.delay;
    // DO THE THINGS
    ctrl.meas_pol = ctrl.meas_cl - ctrl.meas_dm;
    ctrl.mutex.unlock();
}

void reconstruct_modes()
{
    ctrl.mutex.lock();
    // the reconstructed modes are the matrix-vector product of the
    // reconstructor matrix (meas_to_modes) and the pseudo-open loop
    // measurements
    ctrl.modes_pol = ctrl.meas_to_modes * ctrl.meas_pol;
    ctrl.mutex.unlock();
}

void filter_modes()
{
    ctrl.mutex.lock();
    // TODO this one requires some care
    // DO THE THINGS
    ctrl.modes_filt = -ctrl.modes_pol;
    ctrl.mutex.unlock();
}

void project_com()
{
    ctrl.mutex.lock();
    // project the filtered modes to the command space and add the command offset
    ctrl.com_ctrl = (ctrl.modes_to_com * ctrl.modes_filt) + ctrl.com_offset;
    ctrl.mutex.unlock();
}

void inject_disturb()
{
    ctrl.mutex.lock();
    // add the next disturbance buffer element to the command vector
    ctrl.com_raw = ctrl.com_ctrl + ctrl.com_dist_buffer.col(cnt % DIST_LEN);
    ctrl.mutex.unlock();
}

void shm_write()
{
    ctrl.mutex.lock();
    // write to the dm shm
    for (size_t i = 0; i < N_ACTUATORS; i++)
    {
        DM_low.array.D[i] = ctrl.com_raw[i];
    }
    ctrl.mutex.unlock();
}

void inject_dm_signal()
{
    ctrl.mutex.lock();
    ctrl.meas_dm = ctrl.com_to_meas * ctrl.com_ctrl;
    ctrl.mutex.unlock();
}