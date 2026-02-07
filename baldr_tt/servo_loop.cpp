#include "baldr_tt.h"
//#define PRINT_TIMING
//#define PRINT_TIMING_ALL
//#define DEBUG
//#define DEBUG_FILTER6
#define DARK_OFFSET 1000
#define DM_MAX_R 5.0

// Thresholds for fringe tracking (now variables)
double flux_threshold = 1000.0;

using namespace std::complex_literals;

long unsigned int cnt=0, cnt_since_init=0;
long unsigned int nerrors=0;
int sz=0;
int *im_boxcar[N_BOXCAR];
double *window, *subim;
double *im_av, *im_plus, *im_minus, *norm_imsub;
float *im_plus_sum, *im_minus_sum;
std::mutex im_mutex; 
TTMet_save ttmet_save;   

void set_dm_tilt_foc(double tx_in, double ty_in, double focus){
#ifdef SIMULATE
    return;
#endif
    double r2=0.0;
    // Rotate the input tip/tilt by the current rotation matrix. 
    // This allows us to correct for any rotation between the DM and the image.
    double tx = control_u.R(0,0)*tx_in + control_u.R(0,1)*ty_in;
    double ty = control_u.R(1,0)*tx_in + control_u.R(1,1)*ty_in;
    // Set DM tip/tilt and focus terms.
    for (int j=0; j<12; j++)
        for (int i=0; i<12; i++)
        {
            r2 = (i-5.5)*(i-5.5)+(j-5.5)*(j-5.5);
            if (r2 > DM_MAX_R*DM_MAX_R) r2 = DM_MAX_R*DM_MAX_R;
            DM_low.array.D[12*j+i] = ty * (i - 5.5) / DM_MAX_R + 
                tx * (5.5 - j) / DM_MAX_R + focus * (1.0 - 2*r2/DM_MAX_R/DM_MAX_R);
        }
    ImageStreamIO_sempost(&master_DM, 1);
}

// Initialise variables and arrays on startup
void initialise_servo(){
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
    im_av = (double*) malloc(sizeof(double) * sz * sz);
    im_plus = (double*) malloc(sizeof(double) * width * width);
    im_minus = (double*) malloc(sizeof(double) * width * width);
    im_plus_sum = (float*) malloc(sizeof(float) * width * width);
    im_minus_sum = (float*) malloc(sizeof(float) * width * width);
    window = (double*) malloc(sizeof(double) * width * width);
    subim = (double*) malloc(sizeof(double) * width * width);
    norm_imsub = (double*) malloc(sizeof(double) * width * width);
     // Initialise the window to a super-Gaussian with a 1/e^2 width equal to the image size.
    int ssz = (int)width;
    for (int ii=0; ii<ssz; ii++) {
        for (int jj=0; jj<ssz; jj++) {
            double temp = ((double)(ii - ssz / 2) * (double)(ii - ssz / 2) +
            (double)(jj - ssz / 2) * (double)(jj - ssz / 2)) /
            (double)(ssz / 2) / (double)(ssz / 2);
            window[ii*width + jj] = std::exp(-temp*temp);
        }
    }

    // Set these images to zero.
    for (int j=0;j<sz*sz;j++) im_av[j]=0;
    for (int j=0;j<width*width;j++){
        im_plus[j]=0;
        im_minus[j]=0;
        subim[j]=0;
        im_plus_sum[j]=0;
        im_minus_sum[j]=0;
        norm_imsub[j]=0;
    } 

    // Allocate memory for the boxcar averages and set to zero.
    for (int i=0;i<N_BOXCAR;i++){
        im_boxcar[i] = (int*) malloc(sizeof(int) * sz * sz);
        for (int j=0;j<sz*sz;j++){
            im_boxcar[i][j]=0;
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
static inline void catch_up_with_sem(IMAGE* img, int semid) {
    // keep grabbing until there are no more pending posts
    while (ImageStreamIO_semtrywait(img, semid) == 0) { /* nothing just do it*/; }
}

// The main AO servo loop
void servo_loop(){
    double add_tx, add_ty;
    timespec now;
#ifdef DEBUG_ALL
    timespec now_all, then_all;
#endif
    initialise_servo();
    cnt = subarray.md->cnt0;
    catch_up_with_sem(&subarray, 2);
    while(servo_mode != SERVO_STOP){
        cnt_since_init++; //This should "never" wrap around, as a long int is big.
        // See if there was a semaphore signalled for the next frame to be ready in K1 and K2
        ImageStreamIO_semwait(&subarray, 2);
        // If we are here, then a new frame is available in both K1 and K2. 
        // Check that there has not been a counting error.
        if(subarray.md->cnt0 == cnt){
            std::cout << "FT: Semaphore signalled but no new frame" << std::endl;
            nerrors++;
            continue;
        }
        // Check for missed frames
        if (subarray.md->cnt0 > cnt+2){
            std::cout << "Missed frames! Image: " << subarray.md->cnt0 << " Servo: " << cnt << std::endl;
            // Catch up!
            catch_up_with_sem(&subarray, 2);
            cnt = subarray.md->cnt0 - 1;
            nerrors++;
        }
        cnt++;
#ifdef PRINT_TIMING
        timespec then;
        clock_gettime(CLOCK_REALTIME, &then);
#endif
        // Copy the data from the IMAGE subarray to the subimage.
        for (int ii=0; ii<width; ii++) {
            for (int jj=0; jj<width; jj++) {
                int y = settings.s.py - width/2 + ii;
                int x = settings.s.px - width/2 + jj;
                subim[ii*width + jj] = (double)(subarray.array.SI32[y*sz + x]-DARK_OFFSET);
            }
        }
        // Compute the weighted flux within +/- width/2 of the current (px, py) position.
        rt_status.mutex.lock();
        rt_status.s.flux=0;
        rt_status.s.tx = 0;
        rt_status.s.ty = 0;
        for (int ii=0; ii<width; ii++) {
            for (int jj=0; jj<width; jj++) {
                double y = ii - width/2;
                double x = jj - width/2;
                rt_status.s.flux += window[ii*width + jj] * subim[ii*width + jj];
                rt_status.s.tx += window[ii*width + jj] * subim[ii*width + jj] * x;
                rt_status.s.ty += window[ii*width + jj] * subim[ii*width + jj] * y;
            }
        }
        rt_status.s.tx /= rt_status.s.flux;
        rt_status.s.ty /= rt_status.s.flux;
        // If the flux is above the threshold, compute the new DM settings and update the DM image. 
        // Otherwise, skip the DM update and just wait for the next frame.
        if (rt_status.s.flux > settings.s.flux_threshold) {
            // Compute the new DM settings. For now, just a simple proportional controller on the tip/tilt, and a focus term that is proportional to the flux (this is just to test that the focus term is working).
            add_tx = settings.s.ttg * rt_status.s.tx;
            add_ty = settings.s.ttg * rt_status.s.ty;
        } else {
            add_tx = 0.0;
            add_ty = 0.0;
        }
        rt_status.mutex.unlock();
        control_u.tx = (1-settings.s.ttl) * (control_u.tx - settings.s.ttxo) + add_tx;
        control_u.ty = (1-settings.s.ttl) * (control_u.ty - settings.s.ttyo) + add_ty;
        
        // Based on where we are in the modulation, set the high-order modes to be 
        // either positive or negative - relevant for the next image.
        control_u.ho_ix = (control_u.ho_ix + 1) % HO_CYCLE;
        if (control_u.ho_ix == HO_CYCLE - 1) {
            control_u.ho_sign *= -1;
        }
        
        // Set the DM if we are in the appropriate mode.
        if ((servo_mode == SERVO_HO) || (servo_mode == SERVO_TT)) 
            set_dm_tilt_foc(control_u.tx, control_u.ty, settings.s.focus_offset + settings.s.focus_amp * control_u.ho_sign);
        else
            set_dm_tilt_foc(settings.s.ttxo, settings.s.ttyo, settings.s.focus_offset);

        // Update the saved tip/tilt metrology.
        ttmet_save.mx[ttmet_save.cnt] = control_u.tx;
        ttmet_save.my[ttmet_save.cnt] = control_u.ty;
        ttmet_save.ty[ttmet_save.cnt] = rt_status.s.ty;
        ttmet_save.tx[ttmet_save.cnt] = rt_status.s.tx;
        ttmet_save.cnt = (ttmet_save.cnt + 1) % N_TTMET;

        // Accumulate the im_plus and im_minus images for the high-order modulation.
        // ho_ix=0 : last frame invalid.
        // ho_ix=1 : last frame valid
        // ho_ix=2 : last frame valid 
        // ...
        // ho_ix = (HO_CYCLE-1) : last frame valid. Just sent changed focus.

        im_mutex.lock();
        if (control_u.ho_ix > 0) {
            if (control_u.ho_sign > 0) {
                // Clear the plus image at the start of the plus phase.
                if (control_u.ho_ix==1) for (int j=0;j<width*width;j++) im_plus[j]=0; 
                for (int j=0;j<width*width;j++) {
                    im_plus[j] += subim[j];
                }
            } else {
                // Clear the minus image at the start of the minus phase.
                if (control_u.ho_ix==1) for (int j=0;j<width*width;j++) im_minus[j]=0; 
                for (int j=0;j<width*width;j++) {
                    im_minus[j] += subim[j];
                }
            }
        } else {
            // Update the plus or minus sum.
            if (control_u.ho_sign > 0) {
                for (int j=0;j<width*width;j++) im_plus_sum[j] += im_plus[j];
            } else {
                for (int j=0;j<width*width;j++) im_minus_sum[j] += im_minus[j];
            }
        }
        im_mutex.unlock();

        // Are we ready for the high order loop? If so, subtract
        // the inverted im_minus from im_plus.
        if (control_u.ho_ix == (HO_CYCLE-1)){
            im_mutex.lock();
            double sum_both=0;
            for (int j=0;j<width*width;j++) sum_both += im_plus_sum[j] + im_minus_sum[j];
            for (int j=0;j<width*width;j++) {
                norm_imsub[j] = (im_plus_sum[j] - im_minus_sum[width*width - 1 - j]) / sum_both;
            }
            im_mutex.unlock();
            // Here we could compute the high-order modes based on norm_imsub
            //!!! needs a reconstrutor.

            // Multiply the high-order modes by the influence functions to get the DM shape.
            control_u.DM = control_a.influence_functions * control_a.modes;
             // Update the DM image with the new high-order shape.
            for (int i=0; i<N_ACTUATORS; i++) 
                DM_high.array.D[i] = control_u.DM(i);
            ImageStreamIO_sempost(&master_DM, 1);
        }

        // Done with critical parts. Update the boxcar average
        int ix = cnt % N_BOXCAR;
        for (int i=0;i<sz*sz;i++) {
            im_av[i] -= (double)im_boxcar[ix][i];
            im_boxcar[ix][i] = subarray.array.SI32[i] - DARK_OFFSET;
            im_av[i] += im_boxcar[ix][i];
        }
    }
}
