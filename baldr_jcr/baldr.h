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

#define N_MODES 11  // Number of modes to control
#define WIDTH 15  // Number of pixels across subim
#define N_PIXELS WIDTH*WIDTH  // Total number of pixels in subim
#define FILTER_LEN 2  // Max number of taps in IIR filter
#define N_ACTUATORS 144 // Including corners
#define DIST_LEN 10 // Length of disturbance sequence (periodic)

//----- Structures and typedefs------

// Variables for controller.
struct ControlVariables {
    // real-time variables
    Eigen::Matrix<double, N_PIXELS, 1> meas_raw;
    Eigen::Matrix<double, N_PIXELS, 1> meas_cl;
    Eigen::Matrix<double, N_PIXELS, 1> meas_dm;
    Eigen::Matrix<double, N_PIXELS, 1> meas_pol;
    Eigen::Matrix<double, N_MODES, 1> modes_pol;
    Eigen::Matrix<double, N_MODES, 1> modes_filt;
    Eigen::Matrix<double, N_ACTUATORS, 1> com_ctrl;
    Eigen::Matrix<double, N_ACTUATORS, 1> com_raw;

    // dynamically configurable variables
    Eigen::Matrix<double, N_PIXELS, 1> meas_ref;
    double delay;
    Eigen::Matrix<double, N_MODES, N_PIXELS> meas_to_modes;
    Eigen::Matrix<double, N_MODES, FILTER_LEN> filter_coeff_in;
    Eigen::Matrix<double, N_MODES, FILTER_LEN> filter_coeff_out;
    Eigen::Matrix<double, N_ACTUATORS, N_MODES> modes_to_com;
    Eigen::Matrix<double, N_ACTUATORS, 1> com_offset;
    Eigen::Matrix<double, N_ACTUATORS, DIST_LEN> com_dist_buffer;
    Eigen::Matrix<double, N_PIXELS, N_ACTUATORS> com_to_meas;
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
    double flux;
    int64_t nerrors;
    int64_t nlowflux;
    int cnt;
};

// Settings struct for commander
struct Settings
{
    double log, lol, hog, hol, flux_threshold;
    size_t num_lomodes;
    int px, py;
    int servo_mode;
};

enum ServoMode {
    SERVO_STOP=-1,
    SERVO_OPEN,
    SERVO_CLOSED,
};

// variants of commander call results
enum ErrorCode {
    success=0,
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
    Settings settings;
};

// Status including a mutex.
struct RTStatus{
    std::mutex mutex;
    Status status;
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
extern ControlVariables ctrl;
extern uint64_t cnt;

// Images - plus, minus and average
extern std::mutex im_mutex;

// ----- methods to be implemented for servo loop -----
void servo_loop(void);

void read_shm(int &retFlag);

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

void read_shm();
void calibrate_frame();
void compute_pol_meas();
void reconstruct_modes();
void filter_modes();
void project_com();
void inject_disturb();
void shm_write();
void inject_dm_signal();