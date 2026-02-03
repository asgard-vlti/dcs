#include "heimdallr.h"
//#define PRINT_TIMING
#define DARK_OFFSET 1000.0

//------------------------------------------------------------------------------
// Drain any outstanding semaphore posts so that
// the next semwait() really waits for a fresh frame.
//------------------------------------------------------------------------------
static inline void catch_up_with_sem(IMAGE* img, int semid) {
    // keep grabbing until there are no more pending posts
    while (ImageStreamIO_semtrywait(img, semid) == 0) { /* nothing just do it*/; }
}

ForwardFt::ForwardFt(IMAGE * subarray_in) {
    save_dark_frames=false;
    subarray = subarray_in;
    // Sanity check that we actually have a 2D , square image
    if (subarray->md->naxis != 2) {
        throw std::runtime_error("Subarray is not 2D");
    }
    subim_sz = subarray->md->size[0];
    if (subarray->md->size[1] != subim_sz) {
        throw std::runtime_error("Subarray is not square");
    }
    // Allocate memory for the Fourier transform and plan it.
    ft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * subim_sz * (subim_sz / 2 + 1));
    subim = (double*) fftw_malloc(sizeof(double) * subim_sz * subim_sz);
    
    // Also allocate memory for the subimage average and the boxcar frames.
    dark = (double*) fftw_malloc(sizeof(double) * subim_sz * subim_sz);
    subim_av = (double*) fftw_malloc(sizeof(double) * subim_sz * subim_sz);
    for (int ii=0; ii<N_DARK_BOXCAR; ii++) {
        subim_boxcar[ii] = (double*) fftw_malloc(sizeof(double) * subim_sz * subim_sz);
    }
    // Initialise the subimage average and saved dark to zero
    // !!! Once we're really set up, the dark should be set to a previously
    // saved dark calibration frame for our mode !!!
    for (unsigned int ii=0; ii<subim_sz; ii++) {
        for (unsigned int jj=0; jj<subim_sz; jj++) {
            subim_av[ii*subim_sz + jj] = 0.0;
            dark[ii*subim_sz + jj] = 0.0;
        }
    }
    // Initialise the boxcar frames to zero
    for (unsigned int ii=0; ii<subim_sz; ii++) {
        for (unsigned int jj=0; jj<subim_sz; jj++) {
            for (int kk=0; kk<N_DARK_BOXCAR; kk++) {
                subim_boxcar[kk][ii*subim_sz + jj] = 0.0;
            }
        }
    }

    // Create the plan
    plan = fftw_plan_dft_r2c_2d(subim_sz, subim_sz, subim, ft, FFTW_MEASURE);

    // Allocate memory for the subimage
    window = (double*) fftw_malloc(sizeof(double) * subim_sz * subim_sz);
    power_spectrum = (double*) fftw_malloc(sizeof(double) * subim_sz * (subim_sz / 2 + 1));
    //power_spectra[0] = (double*) fftw_malloc(sizeof(double) * subim_sz * (subim_sz / 2 + 1) * MAX_N_PS_BOXCAR);
    for (int ii=0; ii<MAX_N_PS_BOXCAR; ii++) {
        power_spectra[ii] = (double*) fftw_malloc(sizeof(double) * subim_sz * (subim_sz / 2 + 1));
        //power_spectra[ii-1] + subim_sz * (subim_sz / 2 + 1);
    }
    // Initialise the window to a super-Gaussian with a 1/e^2 width equal to the image size.
    // !!! Probably the window should be centered on a half pixel.
    int ssz = (int)subim_sz;
    for (int ii=0; ii<ssz; ii++) {
        for (int jj=0; jj<(int)ssz; jj++) {
            double temp = ((double)(ii - (int)ssz / 2) * (double)(ii - ssz / 2) +
            (double)(jj - ssz / 2) * (double)(jj - ssz / 2)) /
            (double)(ssz / 2) / (double)(ssz / 2);
            window[ii*subim_sz + jj] = std::exp(-temp*temp);
        }
    }
    for (unsigned int ii=0; ii<subim_sz; ii++) {
        // Also initialise the power spectrum array
        for (unsigned int jj=0; jj<subim_sz / 2 + 1; jj++) {
            for (int kk=0; kk<MAX_N_PS_BOXCAR; kk++) {
                power_spectra[kk][ii*(subim_sz/2+1) + jj] = 0.0;
            }
            power_spectrum[ii*(subim_sz/2+1) + jj] = 0.0;
        }
    }
    // Initialise POSIX semaphore for new frame notification
    sem_init(&sem_new_frame, 0, 0);
}

void ForwardFt::set_bad_pixels(std::vector<int> kx, std::vector<int> ky) {
    std::lock_guard<std::mutex> lock(mutex);
    // Just copy the vectors
    bad_pixel_x = kx;
    bad_pixel_y = ky;
}

void ForwardFt::start() {
    thread = std::thread(&ForwardFt::loop, this);
}

void ForwardFt::stop() {
    mode = FT_STOPPING;
    if (thread.joinable()) thread.join();
}

void ForwardFt::loop() {
#ifdef PRINT_TIMING
    timespec now, then;
#endif
    unsigned int ii_shift, jj_shift, szj;
    cnt = subarray->md->cnt0;
    catch_up_with_sem(subarray, 2);
    while (mode != FT_STOPPING) {
        ImageStreamIO_semwait(subarray, 2);
        if (subarray->md->cnt0 != cnt) {
            // Put this here just in case there is a re-start with a new size. Unlikely!
            szj = subim_sz/2 + 1;
            if ((subarray->md->cnt0 > cnt+2)  && (mode == FT_RUNNING)) {
                std::cout << "Missed cam frames: " << subarray->md->cnt0 << " " << cnt << std::endl;
                catch_up_with_sem(subarray,2);
                cnt = subarray->md->cnt0-1;
                nerrors++;
            }
            mode = FT_RUNNING;

            // Copy the data from the IMAGE subarray to the subimage
#ifdef PRINT_TIMING
            clock_gettime(CLOCK_REALTIME, &then);
#endif
            // Set bad pixels to the mean of their neighbours
            for (unsigned int bp=0; bp<bad_pixel_x.size(); bp++){
                    int x = bad_pixel_x[bp];
                    int y = bad_pixel_y[bp];
                    int count=0;
                    double sum=0.0;
                    for (int dx=-1; dx<=1; dx++){
                        for (int dy=-1; dy<=1; dy++){
                            if (dx==0 && dy==0) continue;
                            if (x+dx<0 || x+dx>=(int)subim_sz) continue;
                            if (y+dy<0 || y+dy>=(int)subim_sz) continue;
                            sum += (double)(subarray->array.SI32[(y+dy)*subim_sz + (x+dx)]);
                            count++;
                        }
                    }
                    if (count>0)
                        subarray->array.SI32[y*subim_sz + x] = (int)(sum/(double)count);
            }
            // Now copy the image to the subimage array, applying the window function
            // NB the dark is subtracted elsewhere.
            for (unsigned int ii=0; ii<subim_sz; ii++) {
                for (unsigned int jj=0; jj<subim_sz; jj++) {
                    ii_shift = (ii + subim_sz/2) % subim_sz;
                    jj_shift = (jj + subim_sz/2) % subim_sz;
                    subim[ii_shift*subim_sz + jj_shift] = 
                        ((double)(subarray->array.SI32[ii*subim_sz + jj]) - DARK_OFFSET) //instead of dark[ii*subim_sz + jj_shift]
                            * window[ii*subim_sz + jj];
                }
            }
            // Do the FFT, and then indicate that the frame has been processed
            fftw_execute(plan);

#ifdef PRINT_TIMING
            clock_gettime(CLOCK_REALTIME, &now);
            if (then.tv_sec == now.tv_sec)
                std::cout << "Window and FFT time: " << now.tv_nsec-then.tv_nsec << std::endl;
            then = now;
#endif
            // If the flux is negative, signal a bad frame.
            if (ft[0][0] > 0)
                bad_frame=false;
            else
                {
                    bad_frame=true;
                    cnt++;
                    sem_post(&sem_new_frame);
                    continue;
                }
            // Compute the power spectrum. Other than SNR purposes, this isn't 
            // time critical, but doesn't take long so we can do it here. 
            ps_index = (subarray->md->cnt0 + 1) % MAX_N_PS_BOXCAR;
            for (unsigned int ii=0; ii<subim_sz; ii++) {
                for (unsigned int jj=0; jj<szj; jj++) {
                    int f_ix = ii*szj + jj; // Flattened index
                    power_spectrum[f_ix] -= 
                        power_spectra[ps_index][f_ix]/MAX_N_PS_BOXCAR;
                    power_spectra[ps_index][f_ix] = 
                        ft[f_ix][0] * ft[f_ix][0] +
                        ft[f_ix][1] * ft[f_ix][1];
                    power_spectrum[f_ix] += 
                        power_spectra[ps_index][f_ix]/MAX_N_PS_BOXCAR;
                }
            }
            //std::cout << "PS00: " << power_spectrum[0] << std::endl;

            // Compute the power spectrum bias and instantaneous bias.
            power_spectrum_bias=0;
            power_spectrum_inst_bias=0;
            for (unsigned int ii=subim_sz/2-subim_sz/8; ii<subim_sz/2+subim_sz/8; ii++) {
                for (unsigned int jj=szj - subim_sz/8; jj<szj; jj++) {
                    power_spectrum_bias += power_spectrum[ii*szj + jj];
                    power_spectrum_inst_bias += power_spectra[ps_index][ii*szj + jj];
                }
            }
            power_spectrum_bias /= subim_sz*subim_sz/32; //two squares 1/8 of the subim
            power_spectrum_inst_bias /= subim_sz*subim_sz/32; //two squares 1/8 of the subim


#ifdef PRINT_TIMING
            clock_gettime(CLOCK_REALTIME, &now);
            if (then.tv_sec == now.tv_sec)
                std::cout << "PS time: " << now.tv_nsec-then.tv_nsec << std::endl;
            then = now;
#endif
            // As long as this is the same type as cnt0, it should wrap around correctly
            // The reason it is here and not before power spectrum computation is because we need at
            // lease 1 power spectrum in order for the group delay.
            cnt++;

            // Signal that a new frame is available.
            sem_post(&sem_new_frame);

            // Now we need to boxcar average the dark frames. 
            //!!! This will not even work as UI16 should be SI32
            int ix = cnt % N_DARK_BOXCAR;
            for (unsigned int ii=0; ii<subim_sz; ii++) {
                for (unsigned int jj=0; jj<subim_sz; jj++) {
                    subim_av[ii*subim_sz + jj] -= 
                        subim_boxcar[ix][ii*subim_sz + jj];
                    subim_boxcar[ix][ii*subim_sz + jj] = 
                        (double)(subarray->array.UI16[ii*subim_sz + jj])/N_DARK_BOXCAR;
                    subim_av[ii*subim_sz + jj] +=
                        subim_boxcar[ix][ii*subim_sz + jj];
                }
            }
            if (save_dark_frames) {
                for (unsigned int ii=0; ii<subim_sz; ii++) {
                    for (unsigned int jj=0; jj<subim_sz; jj++) {
                        dark[ii*subim_sz + jj] = subim_av[ii*subim_sz + jj];
                    }
                }
                save_dark_frames = false;
            }

            //std::cout << subarray->name << ": " << cnt << std::endl;
        } else {
            // This shouldn't happen, but if it does, just continue
            std::cout << "FT: Semaphore signalled but no new frame" << std::endl;
            nerrors++;
        }
    }
}
