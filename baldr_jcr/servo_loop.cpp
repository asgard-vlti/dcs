/*
 *   Copyright (c) 2026 Australian National University
 *   All rights reserved.
 */

#include "./baldr.h"
#include "commander/commander.h"
// #define PRINT_TIMING
// #define PRINT_TIMING_ALL
// #define DEBUG
// #define DEBUG_FILTER6
#define DARK_OFFSET 1000
#define DM_MAX_R 5.0

uint64_t cnt = 0, cnt_since_init = 0;
int64_t nerrors = 0;
size_t sz = 0;
int *im_boxcar[N_BOXCAR];
double *window, *subim;
double *im_av, *im_plus, *im_minus, *norm_imsub;
float *im_plus_sum, *im_minus_sum;
std::mutex im_mutex;
TTMet_save ttmet_save;


// Initialise variables and arrays on startup
void initialise_servo() {
    cnt_since_init = 0;
    ttmet_save.cnt = 0;
    // Check the subarray.
    if (subarray.md->naxis != 2) {
        throw std::runtime_error("Subarray is not 2D");
    }
    sz = subarray.md->size[0];
    if (subarray.md->size[1] != sz) {
        throw std::runtime_error("Subarray is not square");
    }
    // Now we know the image size, allocate memory!
    im_av = reinterpret_cast<double *>(malloc(sizeof(double) * sz * sz));
    im_plus = (double *)malloc(sizeof(double) * width * width * (HO_CYCLE-1));  // JCR: changing these to buffers
    im_minus = (double *)malloc(sizeof(double) * width * width * (HO_CYCLE-1));
    im_plus_sum = (float *)malloc(sizeof(float) * width * width);
    im_minus_sum = (float *)malloc(sizeof(float) * width * width);
    // TODO: test with singles here, double seems unnecessary
    window = (double *)malloc(sizeof(double) * width * width);
    subim = (double *)malloc(sizeof(double) * width * width);
    norm_imsub = (double *)malloc(sizeof(double) * width * width);
    // Initialise the window to a super-Gaussian with a 1/e^2 width equal to the image size.
    size_t ssz = width;
    for (size_t ii = 0; ii < ssz; ii++)
    {
        for (size_t jj = 0; jj < ssz; jj++)
        {
            double temp = ((double)(ii - ssz / 2) * (double)(ii - ssz / 2) +
                           (double)(jj - ssz / 2) * (double)(jj - ssz / 2)) /
                          (double)(ssz / 2) / (double)(ssz / 2);
            window[ii * width + jj] = std::exp(-temp * temp);
        }
    }

    // Set these images to zero.
    for (size_t j = 0; j < sz * sz; j++)
        im_av[j] = 0;
    for (size_t j = 0; j < width * width; j++)
    {
        im_plus[j] = 0;
        im_minus[j] = 0;
        subim[j] = 0;
        im_plus_sum[j] = 0;
        im_minus_sum[j] = 0;
        norm_imsub[j] = 0;
    }

    // Allocate memory for the boxcar averages and set to zero.
    for (size_t i = 0; i < N_BOXCAR; i++)
    {
        im_boxcar[i] = (int *)malloc(sizeof(int) * sz * sz);
        for (size_t j = 0; j < sz * sz; j++)
        {
            im_boxcar[i][j] = 0;
        }
    }

    // Initialise the control_u and control_a structures to zero.
    control_u.tx = 0.0;
    control_u.ty = 0.0;
    control_u.ho_ix = 0;
    control_u.ho_sign = 1;
    control_u.DM.setZero();
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
    while (settings.s.servo_mode != SERVO_STOP)
    {
        cnt_since_init++; // This should "never" wrap around, as a long int is big.
        
        // See if there was a semaphore signalled for the next frame to be ready in K1 and K2
        ImageStreamIO_semwait(&subarray, 2);
        
        // If we are here, then a new frame is available in both K1 and K2.
        // Check that there has not been a counting error.
        if (subarray.md->cnt0 == cnt)
        {
            info("FT: Semaphore signalled but no new frame");
            nerrors++;
            continue;
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

        // Copy the data from the IMAGE subarray to the subimage.
        for (size_t ii = 0; ii < width; ii++)
        {
            for (size_t jj = 0; jj < width; jj++)
            {
                int y = settings.s.py - width / 2 + ii;
                int x = settings.s.px - width / 2 + jj;
                subim[ii * width + jj] = (double)(subarray.array.SI32[y * sz + x] - DARK_OFFSET);
            }
        }

        
        
        //// Compute the weighted flux within +/- width/2 of the current (px, py) position.
        // lock the status mutex to be able to read/write real time variables
        rt_status.mutex.lock();

        // zero flux and tip/tilt
        rt_status.s.flux = 0;
        rt_status.s.tx = 0;
        rt_status.s.ty = 0;

        double x, y, weighted_intensity;
        for (size_t ii = 0; ii < width; ii++)
        {
            y = ii - width/2;
            for (size_t jj = 0; jj < width; jj++)
            {
                x = jj - width/2;
                weighted_intensity = window[ii*width + jj] * subim[ii*width + jj];
                rt_status.s.tx += weighted_intensity * x;
                rt_status.s.ty += weighted_intensity * y;
                rt_status.s.flux += weighted_intensity;
            }
        }
        rt_status.mutex.unlock();
        
        // Here's the logic with the HO_CYCLE / ho_ix idea. Note that always HO_CYCLE>=2
        // First, we compute the ho_ix, which is between 0..2*HO_CYCLE.
        // The value of this index determines what we do with the current image:
        //       0      -> We've just applied a positive defocus, so ignore this frame
        //                 while the DM settles.
        //       1      -> The 1st valid positively defocussed image
        //       2      -> The 2nd valid ... 
        //      ...     ->      ...
        //   HO_CYCLE-1 -> The HO_CYCLE'th positively defocussed image
        //    HO_CYCLE  -> We've just applied a negative defocus, so ignore this frame
        //                 while the DM settles.
        //   HO_CYCLE+1 -> The 1st valid negative defocussed image
        //   HO_CYCLE+2 -> The 2nd valid ... 
        //      ...     ->      ...
        // 2*HO_CYCLE-1 -> The HO_CYCLE'th negative defocussed image
        // 2*HO_CYCLE == 0, so repeat the loop.

        control_u.ho_ix = (control_u.ho_ix + 1) % (2 * HO_CYCLE);
        // If the flux is above the threshold, compute the new DM settings and update the DM image.
        // Otherwise, skip the DM update and just wait for the next frame.
        if (rt_status.s.flux > settings.s.flux_threshold)
        {
            // ALWAYS compute the measurement signal, even if no loops are closed.
            
            // First, we update the appropriate image buffer.
            if (control_u.ho_ix == 0 || control_u.ho_ix == HO_CYCLE) {
                // skip this frame since the focus offset hasn't settled yet
            } else if (control_u.ho_ix < HO_CYCLE) {
                // positive defocus is applied, copy im to appropriate buffer
                size_t offset = (control_u.ho_ix-1) * width * width;
                for (size_t i = 0; i < width*width; i++) {
                    im_plus[i+offset] = subim[i];
                }
            } else {
                size_t offset = (control_u.ho_ix-1-HO_CYCLE) * width * width;
                for (size_t i = 0; i < width*width; i++) {
                    im_minus[i+offset] = subim[i];
                }
            }
            
            // Second, we use all images in both buffers to compute the latest
            // command
            double sum_flux = 0.0;
            for (size_t i = 0; i < width*width; i++){
                for (size_t k = 0; k < HO_CYCLE; k++){
                    sum_flux += im_plus[k*width*width+i] + im_minus[(k+1)*width*width - i - 1];
                    norm_imsub[i] = im_plus[k*width*width+i] - im_minus[(k+1)*width*width - i - 1];
                }
            }
            for (size_t i = 0; i<width*width; i++) {
                norm_imsub[i] /= sum_flux;
            }

            // Third, if we are at least in TT mode, then apply a TT correction
            if (settings.s.servo_mode >= SERVO_TT) {
                // update LO modes based on measurement
                for (size_t i = 0; i < N_MODES; i++) {
                    control_a.modes[i] = 0.0;
                    for (size_t j = 0; j < width*width; j++) {
                        control_a.modes[i] += control_a.reconstructor[i*N_MODES+j] * norm_imsub[j];
                    }
                }


                // Command the LO DM if we are in the appropriate mode.
                //   DM_low.array.D_i = influence_ij * command_j
                for (size_t i = 0; i < N_ACTUATORS; i++) {
                    double command = 0.0;
                    for (size_t j = 0; j < N_MODES; j++) {
                        command += control_a.influence_functions(i, j) * control_a.modes(j);
                    }
                    DM_low.array.D[i] = command;
                }


                // update the master DM semaphore, I guess to trigger the recalculation of the sum
                // of DM images
                // TODO: why master dm semaphore and not the one we wrote to?
                ImageStreamIO_sempost(&master_DM, 1);
            }
            // Update the saved LO metrology.    

            // lock sensor image mutex
            im_mutex.lock();
            
            for (size_t i = 0; i<width*width; i++){
                im_plus_sum[i] = subarray.array.F[i];
            }

            // do measurement calculations
            
            // unlock sensor image mutex
            im_mutex.unlock();
            
            if (settings.s.servo_mode >= SERVO_HO) {
                // update HO DM signal based on measurement
                
                




                // post semaphore on master DM image
                ImageStreamIO_sempost(&master_DM, 1);
            }
        }
        // -- done with critical parts
        
        // update statistics

    }

    // servomode set to stop, clean up anything on exit here.
}
