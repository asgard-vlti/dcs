#include "baldr_tt.h"
//#define PRINT_TIMING
//#define PRINT_TIMING_ALL
//#define DEBUG
//#define DEBUG_FILTER6
#define DARK_OFFSET 1000

// Thresholds for fringe tracking (now variables)
double flux_threshold = 100.0;

using namespace std::complex_literals;

long unsigned int cnt=0, cnt_since_init=0;
long unsigned int nerrors=0;
int sz=0;
int *im_boxcar[N_BOXCAR];

void set_dm_tilt_foc(int tx, int ty, int focus){
#ifdef SIMULATE
    return;
#endif
    // Set DM tip/tilt and focus terms.
    for (int j=0; j<12; j++)
        for (int i=0; i<12; i++)
                DM.array.D[12*j+i] = 0.0;
    ImageStreamIO_sempost(&master_DM, 1);
}

// Initialise variables and arrays on startup
void initialise_servo(){
    cnt_since_init = 0;
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

    // Set these images to zero.
    for (int j=0;j<sz*sz;j++) im_av[j]=0;
    for (int j=0;j<width*width;j++){
        im_plus[j]=0;
        im_minus[j]=0;
    } 

    // Allocate memory for the boxcar averages and set to zero.
    for (int i=0;i<N_BOXCAR;i++){
        im_boxcar[i] = (int*) malloc(sizeof(int) * sz * sz);
        for (int j=0;j<sz*sz;j++){
            im_boxcar[i][j]=0;
        }
    }
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

        // Done with critical parts. Update the boxcar average
        int ix = cnt % N_BOXCAR;
        for (int i=0;i<sz*sz;i++) {
            im_av[i] -= (double)im_boxcar[ix][i];
            im_boxcar[ix][i] = subarray.array.SI32[i] - DARK_OFFSET;
            im_av[i] += im_boxcar[ix][i];
        }
    }
}
