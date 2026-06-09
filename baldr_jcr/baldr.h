#pragma once

//
// /home/asg/.conda/envs/asgard/lib/python3.10/site-packages/asgard_lab_DM_tools/asgard_lab_MDM_controller.py 
// 
#include <complex> 
#include <fftw3.h>
#include <ImageStreamIO.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <atomic>
#define TOML_HEADER_ONLY 0
#include <toml.hpp>
#include <mutex>
#include <thread>
#include <Eigen/Dense>
#include <fmt/core.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <zmq.hpp>
#include <chrono>
#include <fitsio.h>
#include <semaphore.h>
#include <nlohmann/json.hpp>

//----------Defines-----------
//#define SIMULATE

#define N_MODES 11
#define WIDTH 15
#define N_PIXELS WIDTH*WIDTH
#define N_ACTUATORS 144 // Including corners.
#define N_BOXCAR 16
#define N_TTMET 1000
#define HO_CYCLE 3 // A high-order cycle. 

#define FT_STARTING 0
#define FT_RUNNING 1
#define FT_STOPPING 2

//----- Structures and typedefs------
typedef std::complex<double> dcomp;

// Variables for actuation.
struct ControlU{
    double tx, ty;
    int ho_sign;
    int ho_ix;
    Eigen::Matrix<double, N_ACTUATORS, 1> DM;
};

// This is our knowledge of the DM modes
struct ControlA{
    Eigen::Matrix<double, N_MODES, 1> modes;
    Eigen::Matrix<double, N_ACTUATORS, N_MODES> influence_functions;
    double reconstructor[N_MODES*N_PIXELS];
};

struct TTMet_save{
    std::mutex mutex;
    double tx[N_TTMET], ty[N_TTMET], mx[N_TTMET], my[N_TTMET];
    unsigned int cnt;
};

//-------Commander structs-------------
// An encoded 2D image in row-major form.
struct EncodedImage
{
    unsigned int szx, szy;
    std::string type;
    std::string message;
};

// The status, encoded as std::vector<double> for 
// key variables.
struct Status
{
    double flux, tx, ty;
    int cnt;
};

// Settings struct for commander
struct Settings
{
    double ttg, ttl, hog, hol, focus_amp, flux_threshold;
    double gauss_hwidth;
    double ttxo, ttyo, focus_offset;
    int px, py;
    int32_t dark_offset;
    int servo_mode;
};

enum ServoMode {
    SERVO_STOP=-1,
    SERVO_OFF,
    SERVO_TT,
    SERVO_HO,
};

struct TTMet
{
    std::vector<double> tx, ty, mx, my;
    unsigned int cnt;
};

struct ImAvgs
{
    int width;
    std::string im_plus_sum_encoded;
    std::string im_minus_sum_encoded;
};

// variants of commander call results
enum ErrorCode {
    success,
    failure,
};

// Result of commander call
struct Result
{
    ErrorCode status_code;
    nlohmann::json data;
};

#define SUCCESS(json) Result{ErrorCode::success,json}
#define FAILURE(json) Result{ErrorCode::failure,json}

//-------End of Commander structs------

// Settings including a mutex.
struct CtrlSettings{
    std::mutex mutex;
    Settings s;
};

// Status including a mutex.
struct RTStatus{
    std::mutex mutex;
    Status s;
};

// -------- Extern global definitions ------------
extern IMAGE DM_low;
extern IMAGE DM_high;
extern IMAGE master_DM;
extern IMAGE subarray;
// The statit initial input parameters
extern toml::table config;

// Parameters that really don't change after startup.
extern size_t beam, width, sz;

// Servo parameters. These are the parameters that will be adjusted by the commander
extern CtrlSettings settings;
extern RTStatus rt_status;
extern ControlU control_u;
extern ControlA control_a;
extern uint64_t cnt;

// Images - plus, minus and average
extern double *im_av, *im_plus, *im_minus;
extern float *im_plus_sum, *im_minus_sum;
extern std::mutex im_mutex;
extern TTMet_save ttmet_save;

// ----- methods to be implemented for servo loop -----
void servo_loop(void);

// ----- commander methods -----

// load a reconstructor from filename
Result load_reconstructor(std::string filename);

// Set the servo mode
Result set_servo_mode(std::string mode);

// Set the tip-tilt gain.
Result set_ttg(double gain);

// Set the tip/tilt leaky integrator term
Result set_ttl(double leak);

// Set the high order gain
Result set_hog(double gain);

// Set the high order leaky integrator term
Result set_hol(double leak);

// Set the amplitude of the focus term
Result set_focus_amp(double focus);

Result set_flux_threshold(double val);

// Set the number of pixels in x and y
Result set_pxy(size_t px_new, size_t py_new);

Result set_tt_offset(double x, double y);

Result set_focus_offset(double offset);

Result get_status();

Result get_settings();

// Set the pixels to ignore in reconstruction
Result set_bad_pixels(std::vector<int> x, std::vector<int> y);

// ??
Result zero_tt();

// Get the saved tip-tilt metrology since the last counter value
Result get_ttmet(unsigned int last_cnt);

// Poke a mode and get the average image back
Result poke_mode(int mode_ix, double amplitude);