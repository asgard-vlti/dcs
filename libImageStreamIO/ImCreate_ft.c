/*
 * Example code to write image in shared memory
 *
 * compile with:
 * gcc ImCreate_ft.c ImageStreamIO.c -o imft -lm -lpthread -lfftw3
 *
 * Required files in compilation directory :
 * ImCreate_test.c   : source code (this file)
 * ImageStreamIO.c   : ImageStreamIO source code
 * ImageStreamIO.h   : ImageCreate function prototypes
 * ImageStruct.h     : Image structure definition
 *
 * EXECUTION:
 * ./imft --socket tcp://*:7501 
 *
 * Creates an image imtest00 in shared memory
 * Updates the image every ~ 10ms, forever...
 *
 */
#define SZ 32
#define NWAVE 11 // Number of wavelengths per bandpass. Must be odd.
#define DT_US 2000              // Wait 2ms = 2000 microseconds
#define ATM_DAMPING 1e-2      // Damping factor for atmospheric delay evolution


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <complex>
#include <fftw3.h>
#include <iostream>

//-------Commander structs-------------

// The status. 
struct Status
{
    std::string cam_status;
    unsigned int skipped_frames = 0, nbreads=1, tsig_len=2;
    bool shm_error=false;
    double fps=200.0;
};

//-------End of Commander structs------


#include <commander/commander.h>
#include "commander_structs.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include "ImageStreamIO.h"
using namespace std::complex_literals;

/* [geometry] 
hole_diameter   = 9e-3
beam_x          = [-15.6e-3,15.6e-3,0,0]
beam_y          = [-9e-3,-9e-3,0,18e-3]
pix = 24.0
 */

const float hole_radius = 1.65;      // Radius of the hole in pupil pixels
//const float hole_x[4] = {-3.46*hole_radius, 0,0,3.46*hole_radius}; // Hole x center in pupil pixels
//const float hole_y[4] = {-2*hole_radius, 4*hole_radius, 0 , -2*hole_radius}; // Hole y center in pupil pixels
std::vector <double> delays = {0.0, 0.0, 0.0, 0.0}; // Delay line positions in microns, for the 4 telescopes
std::vector <double> atm_delays = {0,0,0,0}; // Atmospheric delays
bool keepgoing = true;
const float hole_x[4] = {-0.00792, -0.0085, 0.0,0.02130};
const float hole_y[4] = {0.019,-0.01484, 0.0, -0.00015};
const float s = SZ*24/2.1;

// Globals that can be changed by commander
double atm_delta = 0.15; // NB get to 100 times this in 10,000 iterations, with damping factor of 0.0001

int make_image(IMAGE *imarray, fftw_complex *pupil, fftw_complex *image, fftw_plan plan, double cwave, double bw, double flux_scale)
{
    double x,y, rnoise;                    // Image column and row indices
    std::complex<double> hole_phasors[4];              // Phasors for the holes
    std::complex<double> val;              // Value of the Fourier transform at a given pixel
    int ix;
    imarray->md->write = 1;         // Poor-man's mutex when writing
    unsigned int* dotUI = imarray->array.UI32;
    // Fill the array with zeros
    for (int ii=0;ii<SZ*SZ;ii++){
        rnoise = rand()/(float)RAND_MAX - 0.5;
        *(dotUI++) = 1000 + rnoise*100;
    }
    // Loop through subwavelengths
    for (int wix=-NWAVE/2;wix<=NWAVE/2;wix++){
        double wave = cwave + (double)wix/(NWAVE-1)*bw;
        // Make our hole phasors
        for (int kk=0; kk<4; kk++)
            hole_phasors[kk] = std::exp(2i*M_PI*(delays[kk] + atm_delays[kk])/wave);

        // Fill the pupil with the holes
        for(int jj=0; jj<SZ; jj++)            // loop rows
        {
            y = (((jj + SZ/2) % SZ) - SZ/2.0);
            for(int ii=0; ii<SZ; ii++)     // loop columns
            {
                x = (((ii + SZ/2) % SZ) - SZ/2.0);
                pupil[jj*SZ + ii][0] = 0.0;
                pupil[jj*SZ + ii][1] = 0.0;
                for (int kk=0; kk<4; kk++)
                {
                    double dx = x-s*hole_x[kk]*2.05/wave;
                    double dy = y-s*hole_y[kk]*2.05/wave;
                    if ((dx*dx+ dy*dy) < 1.5*1.5*2.05*2.05/wave/wave)
                    {
                        pupil[jj*SZ + ii][0] += hole_phasors[kk].real();
                        pupil[jj*SZ + ii][1] += hole_phasors[kk].imag();
                    }
                }
            }
        }
        // Use our FFT plan to transform the pupil to the image
        fftw_execute(plan);

        // Now take the square of the electric field and copy to the image
        // ->array is union; ->array.UI32 is UI32 pointer to image
        dotUI = imarray->array.UI32;
        for (int ii=0; ii<SZ; ii++)
        {
            for (int jj=0; jj<SZ; jj++)
            {
                ix = ((ii + SZ/2) % SZ)*SZ + (jj + SZ/2) % SZ;
                val.real(image[ix][0]);
                val.imag(image[ix][1]);
                *(dotUI++) += (int)std::norm(val)*flux_scale/NWAVE;
            }
        }
    }
    imarray->md->write=0;

    // Post all semaphores (index = -1)
    ImageStreamIO_sempost(imarray, -1);

    imarray->md->cnt0++;
    imarray->md->cnt1++;

    return 0;
}

void* simulate_heimdallr(void *arg)
{
    fftw_complex *pupil_K1, *image_K1, *pupil_K2, *image_K2;
    fftw_plan plan_K1, plan_K2;
    IMAGE imarray[2];              // pointer to array of images
    uint32_t imsize[2] = { SZ, SZ }; // image size is 512 x 512
    long naxis = sizeof(imsize) / (sizeof *imsize);  // # of axes

    // Data type; see file ImageStruct.h for list of supported types
    uint8_t atype = _DATATYPE_INT32;

    int shared = 1;                // 1 if image in shared mem
    int NBkw = 10;                 // number of keywords allowed

    // allocate the fftw arrays
    pupil_K1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * SZ * SZ);
    pupil_K2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * SZ * SZ);
    image_K1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * SZ * SZ);
    image_K2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * SZ * SZ);
    plan_K1 = fftw_plan_dft_2d(SZ, SZ, pupil_K1, image_K1, FFTW_FORWARD, FFTW_MEASURE);
    plan_K2 = fftw_plan_dft_2d(SZ, SZ, pupil_K2, image_K2, FFTW_FORWARD, FFTW_MEASURE);

    // create an image in shared memory
    ImageStreamIO_createIm(imarray, "shei_k1", naxis, imsize, atype, shared, NBkw);
    ImageStreamIO_createIm(imarray+1, "shei_k2", naxis, imsize, atype, shared, NBkw);


    // Writes an image, with random noise on top.
    while (keepgoing)
    {
        make_image(imarray, pupil_K1, image_K1, plan_K1, 2.05,0.2, 1.0);
        make_image(imarray+1, pupil_K2, image_K2, plan_K2, 2.15,0.2, 0.6);
        for (int kk=0;kk<4;kk++){
            atm_delays[kk] *= 1 - ATM_DAMPING;
            atm_delays[kk] += (std::rand()/(double)RAND_MAX - 0.5) * atm_delta * 3.46; //2 * sqrt(3)
        }
        usleep(DT_US);           // Wait 
    }
    return NULL;
}

std::string simrmn(std::vector<double> delays_in) {
    // This function simulates the RMN delay by applying a phase shift based on the input delay.
    // The delay is in microns, and we convert to radians using the K1 wavelength.
    delays = delays_in;
    // print new delays
    std::cout << "Updated delays: ";
    for (size_t i = 0; i < delays.size(); i++) {
        std::cout << delays[i] << " ";
    }
    std::cout << std::endl;
    return "OK";
}

std::string set_atm_delta(double delta) {
    if (delta <= 0.0 || delta > 1.0) return "ERROR: ATM_DELTA out of range (0.0 to 1.0 microns)";
    atm_delta = delta;
    return "OK";
}

Status get_status(){
    Status status;
    status.cam_status = "SIMULATED";
    status.skipped_frames = 0;
    status.nbreads = 1;
    status.tsig_len = 2;
    status.shm_error = false;
    status.fps = 1000000/DT_US;
    return status;
}

COMMANDER_REGISTER(m)
{
    using namespace commander::literals;

    // You can register a function or any other callable object as
    // long as the signature is deductible from the type.
    m.def("simrmn", simrmn, "Simulate the RMN delay by applying a phase shift based on the input delay.", 
        "delays"_arg);
    m.def("set_atm_delta", set_atm_delta, "Set the amplitude of atmospheric delay variations.", "delta"_arg);
    m.def("status", get_status, "Get the current status of the system");
}

int main(int argc, char* argv[])
{
    pthread_t sim_thread;
    pthread_create(&sim_thread, NULL, simulate_heimdallr, NULL);

    // Initialize the commander server and run it
    commander::Server s(argc, argv);
    s.run();
    keepgoing = false;
    pthread_join(sim_thread, NULL);
    return 0;
}


