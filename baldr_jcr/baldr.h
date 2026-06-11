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

#define SUCCESS(json) Result{ErrorCode::success,json}
#define FAILURE(json) Result{ErrorCode::failure,json}

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
    std::mutex mutex;

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

struct MeasBase64
{
    std::string meas;
};

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

// ----- commander methods -----

// load a reconstructor from filename
// Result load_reconstructor(std::string filename);

// Set the servo mode
Result reset_ctrl();

Result set_servo_mode(std::string mode);

Result set_flux_threshold(double val);

Result set_pxy(size_t px_new, size_t py_new);

Result get_status();

Result get_settings();

// Poke a mode and get the average image back
// Result poke_mode(int mode_ix, double amplitude);

void read_shm();
void calibrate_frame();
void compute_pol_meas();
void reconstruct_modes();
void filter_modes();
void project_com();
void inject_disturb();
void shm_write();
void inject_dm_signal();

#define DEF_READ_CTRL_PARAM(PARAM_NAME, DESCRIPTION, NROWS, NCOLS, DATATYPE) \
Result read_##PARAM_NAME(std::string filename) { \
  info("Loading " #DESCRIPTION); \
  FITS_TO_MATRIX(filename, PARAM_NAME, NROWS, NCOLS, DATATYPE); \
  info("Updated " #DESCRIPTION " in controller"); \
  return SUCCESS(status); \
}

#define FITS_TO_MATRIX(FILENAME, PARAM_NAME, NROWS, NCOLS, DATATYPE) \
  fitsfile *fptr; \
  int status = 0; \
  long fpixel[2] = {1, 1}; \
  info("opening file %s", FILENAME.c_str()); \
  fits_open_file(&fptr, FILENAME.c_str(), READONLY, &status); \
  if (status != 0) { \
    return FAILURE(status); \
  } \
  fits_read_pix(fptr, T##DATATYPE, fpixel, NROWS*NCOLS, NULL, ctrl.PARAM_NAME.data(), NULL, &status); \
  if (status != 0) { \
    return FAILURE(status); \
  }

#define LOAD_FROM_FILE(PARAM_NAME, DESCRIPTION) { \
  if (config[#PARAM_NAME].is_string()) { \
    int status = read_##PARAM_NAME(config[#PARAM_NAME].value_or("")).status_code; \
    if (status != 0) { \
        error("Failed to read file specified by config[" #PARAM_NAME "], code %d", status); \
        return status;\
    } \
  } else { \
    warn(#DESCRIPTION " (" #PARAM_NAME ") is not provided in config."); \
    info("Setting " #PARAM_NAME " to zeros."); \
    ctrl.PARAM_NAME.setZero(); \
  } \
}
