#include "heimdallr.h"
//#define PRINT_TIMING
#define DARK_OFFSET 1000.0

FourierSampling fs;

//------------------------------------------------------------------------------
// Drain any outstanding semaphore posts so that
// the next semwait() really waits for a fresh frame.
//------------------------------------------------------------------------------
static inline void catch_up_with_sem(IMAGE* img, int semid) {
    // keep grabbing until there are no more pending posts
    while (ImageStreamIO_semtrywait(img, semid) == 0) { /* nothing just do it*/; }
}

ForwardFt::ForwardFt(IMAGE * subarray_in) {
    subarray = subarray_in;
    // Sanity check that we actually have a 2D , square image
    if (subarray->md->naxis != 2) {
        throw std::runtime_error("Subarray is not 2D");
    }
    subim_sz = subarray->md->size[0];
    if (subarray->md->size[1] != subim_sz) {
        throw std::runtime_error("Subarray is not square");
    }
    // Check that subim_sz is divisible by 4 - needed for reverse FT
    if (subim_sz % 4 != 0) {
        throw std::runtime_error("Subarray size is not divisible by 4");
    }
    rft_sz = subim_sz/4; 
    // Initialise the filter number.
    size_t lastIndex = strlen(subarray->name) - 1;
    if (subarray->name[lastIndex] == '1') {
        filternum = 1;
    } else if (subarray->name[lastIndex] == '2') {
        filternum = 2;
    } else {
        throw std::runtime_error("Subarray name does not end with 1 or 2 to indicate filter");
    }
    // Allocate memory for the Fourier transform and other variables.
    ft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * subim_sz * (subim_sz / 2 + 1));
    ft_copy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * subim_sz * (subim_sz / 2 + 1));
    subim = (double*) fftw_malloc(sizeof(double) * subim_sz * subim_sz);
    ift_result = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rft_sz * rft_sz);
    ift = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * rft_sz * rft_sz);
    
    // Allocate the memory for the boxcar averaged baseline power, and
    // fill with zeros. We allocate the boxcar arrays as a contiguous block of 
    // memory for each baseline, and then set the pointers.
    // Lots of small mallocs is meant to be avoided for speed.
    for (int ii=0; ii<N_BL; ii++) {
        baseline_power_avg[ii] = (double*) malloc(sizeof(double) * rft_sz * rft_sz);
        for (unsigned int kk=0; kk<rft_sz*rft_sz; kk++)
            baseline_power_avg[ii][kk] = 0.0;
        baseline_power_boxcar[ii][0] = (double*) malloc(sizeof(double) * rft_sz * rft_sz * MAX_N_GD_BOXCAR);
        for (int jj=0; jj<MAX_N_GD_BOXCAR; jj++){
            baseline_power_boxcar[ii][jj] = baseline_power_boxcar[ii][0] + jj * rft_sz * rft_sz ;
            for (unsigned int kk=0; kk<rft_sz*rft_sz; kk++){
                baseline_power_boxcar[ii][jj][kk] = 0.0;
            }
        }
    }

    // Create the plan for the Forward transform and Reverse transform.
    plan = fftw_plan_dft_r2c_2d(subim_sz, subim_sz, subim, ft, FFTW_MEASURE);
    rplan = fftw_plan_dft_2d(rft_sz, rft_sz, ift, ift_result, FFTW_BACKWARD, FFTW_MEASURE);

    // Allocate memory for the subimage
    window = (double*) fftw_malloc(sizeof(double) * subim_sz * subim_sz);
    power_spectrum = (double*) fftw_malloc(sizeof(double) * subim_sz * (subim_sz / 2 + 1));
    for (int ii=0; ii<MAX_N_PS_BOXCAR; ii++) {
        power_spectra[ii] = (double*) fftw_malloc(sizeof(double) * subim_sz * (subim_sz / 2 + 1));
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

    double bl_x, bl_y;
    double pix = config["geometry"]["pix"].value_or(24.0);
    double wave;
    if (filternum==1) {
        wave = config["wave"]["K1"].value_or(2.05);
    }
    else if (filternum==2) {
        wave = config["wave"]["K2"].value_or(2.25);
    } else {
        throw std::runtime_error("Wrong filternum!");
    }
    for (int bl=0; bl<N_BL; bl++){
        // Set the x and y coordinates for extracting flux
        bl_x = config["geometry"]["beam_x"][baseline2beam[bl][1]].value_or(0.0) -
            config["geometry"]["beam_x"][baseline2beam[bl][0]].value_or(0.0);
        bl_y = config["geometry"]["beam_y"][baseline2beam[bl][1]].value_or(0.0) -
            config["geometry"]["beam_y"][baseline2beam[bl][0]].value_or(0.0);
        if (bl_x < 0){
            bl_x = -bl_x;
            bl_y = -bl_y;
            fs.sign[bl] = -1;
        } else fs.sign[bl] = 1;
        if (filternum == 1) {
            fs.x_px_K1[bl] = bl_x * pix / wave * subim_sz;
            fs.y_px_K1[bl] = bl_y * pix / wave * subim_sz;
            if (bl_y < 0){
                fs.y_px_K1[bl] += subim_sz;
            }
        }
        else if (filternum == 2) {
            fs.x_px_K2[bl] = bl_x * pix / wave * subim_sz;
            fs.y_px_K2[bl] = bl_y * pix / wave * subim_sz;
            if (bl_y < 0){
                fs.y_px_K2[bl] += subim_sz;
            }
        } else {
            throw std::runtime_error("Wrong filternum!");
        }
        //std::cout << "Baseline: " << bl << " x_px_K1: " << x_px_K1[bl] << " y_px_K1: " << y_px_K1[bl] << std::endl;
        //std::cout << "Baseline: " << bl << " x_px_K2: " << x_px_K2[bl] << " y_px_K2: " << y_px_K2[bl] << std::endl;
    }
    // Initialise POSIX semaphore for new frame notification and
    // reverse Fourier transforms.
    sem_init(&sem_new_frame, 0, 0);
    sem_init(&sem_reverse_ft_ready, 0, 0);
}

void ForwardFt::set_bad_pixels(std::vector<unsigned int> kx, std::vector<unsigned int> ky) {
    std::lock_guard<std::mutex> lock(mutex);
    // Just copy the vectors
    bad_pixel_x = kx;
    bad_pixel_y = ky;
}

void ForwardFt::start() {
    thread = std::thread(&ForwardFt::loop, this);
    reverse_thread = std::thread(&ForwardFt::reverse_ft, this);
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
        // At this point, subarray->md->cnt0 has been incremented by the camera thread, 
        // and the new frame is available in subarray->array.SI32. 
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
            // Check the write parameter. It really shouldn't be active.
            if (subarray->md->write) {
                std::cout << "FT: Semaphore signalled but write flag is still set. Skipping frame." << std::endl;
                continue;
            }
            // In NDMR mode, the first pixel of the image contains the frame counter. 
            // Data are not valid unless this is less than:
            // control_u.nbreads - 1 - control_u.tsig_len
            if ( (control_u.nbreads > 1) && (subarray->array.SI32[0] > (int)(control_u.nbreads - 1 - control_u.tsig_len)) ) {
            	 bad_frame=true;
                 cnt++;
                 sem_post(&sem_new_frame);
                 continue;
            }


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
            if (ft[0][0] > 0){
                bad_frame=false;
            }
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

            // Copy the Fourier Transform to the place needed for reverse_ft
            // !!! If we were clever, we'd just swap pointers.
            reverse_ft_mutex.lock();
            memcpy(ft_copy, ft, sizeof(fftw_complex) * subim_sz * (subim_sz / 2 + 1));
            reverse_ft_mutex.unlock();

            // Now ready for the reverse_ft.
            sem_post(&sem_reverse_ft_ready);

            //std::cout << subarray->name << ": " << cnt << std::endl;
        } else {
            // This shouldn't happen, but if it does, just continue
            std::cout << "FT: Semaphore signalled but no new frame" << std::endl;
            nerrors++;
        }
    }
}

void ForwardFt::reverse_ft() {
    // This is called by the fringe tracker thread when it is ready for a reverse FT. 
    // It should be called after sem_wait(&sem_reverse_ft_ready).
    // It executes the core code if bad_frame is false.
    int boxcar_index=0, x_px, y_px;
    while (mode != FT_STOPPING) {
        // No counters here. Just go whenever we can!
        sem_wait(&sem_reverse_ft_ready);

        // Copy the relevant pixels into the arrays ready for inverse transform.
        for (int bl=0; bl<N_BL; bl++){
            // Find the relevant pixels for K1 or K2.
            // !!! Obviously a place to speed up if we wanted...
            // could be pre-calculated.
            if (filternum==1) {
                x_px = lround(fs.x_px_K1[bl]) % K1ft->subim_sz;
                y_px = lround(fs.y_px_K1[bl]) % K1ft->subim_sz;
            }
            else if (filternum==2) {
                x_px = lround(fs.x_px_K2[bl]) % K2ft->subim_sz;
                y_px = lround(fs.y_px_K2[bl]) % K2ft->subim_sz;
            } else {
                std::cout << "Wrong filternum! " << std::endl;
                continue;
            }
            // During the copying loop only, we need a lock
            reverse_ft_mutex.lock();
            for (int ii=-2; ii<=2; ii++) {
                for (int jj=-2; jj<=2; jj++) {
                    // If the absolute value of ii and jj is both 2, continue.
                    if (abs(ii)==2 && abs(jj)==2) continue;
                    int ift_ix = (ii+2)*rft_sz + (jj+2);
                    int ft_yix = y_px + jj;
                    int ft_xix = x_px + ii;
                    double sign=1;
                    if (ft_xix < 0){
                        ft_yix = -ft_yix;
                        ft_xix = -ft_xix;
                        sign=-1;
                    }
                    // Now correct ft_yix, which could be negative or more than subim_sz.
                    ft_yix = (ft_yix + subim_sz) % subim_sz;
                    ift[ift_ix][0] = ft_copy[ft_yix*(subim_sz/2+1) + ft_xix][0];
                    ift[ift_ix][1] = ft_copy[ft_yix*(subim_sz/2+1) + ft_xix][1] * sign;
                }
            }
            reverse_ft_mutex.unlock();
            // Now do the FFT
            fftw_execute(rplan);
            
            // Take square modulus and add to the boxcar average.
            // !!! TODO: use settings.s.n_gd_boxcar instead of MAX_N_GD_BOXCAR
            baseline_power_mutex.lock();
            for (unsigned int ii=0; ii<rft_sz*rft_sz; ii++) {
                baseline_power_avg[bl][ii] -= baseline_power_boxcar[bl][boxcar_index][ii]/MAX_N_GD_BOXCAR;
                baseline_power_boxcar[bl][boxcar_index][ii] = 
                    ift_result[ii][0]*ift_result[ii][0] + 
                    ift_result[ii][1]*ift_result[ii][1];
                baseline_power_avg[bl][ii] += baseline_power_boxcar[bl][boxcar_index][ii]/MAX_N_GD_BOXCAR;
            }
            baseline_power_mutex.unlock();

        }            
        boxcar_index = (boxcar_index + 1) % MAX_N_GD_BOXCAR;
    }   
}
