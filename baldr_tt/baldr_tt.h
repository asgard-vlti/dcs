//
// /home/asg/.conda/envs/asgard/lib/python3.10/site-packages/asgard_lab_DM_tools/asgard_lab_MDM_controller.py 
// 
#include <complex> 
#include <fftw3.h>
#include <ImageStreamIO.h>
#include <stdlib.h>
#include <iostream>
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

//----------Defines-----------
//#define SIMULATE

#define N_MODES 10
#define N_ACTUATORS 144 // Including corners.
#define N_BOXCAR 16
#define N_TTMET 1000
#define HO_CYCLE 3 // A high-order cycle. 

#define FT_STARTING 0
#define FT_RUNNING 1
#define FT_STOPPING 2

#define SERVO_OFF 0
#define SERVO_TT 1
#define SERVO_HO 2
#define SERVO_STOP -1

//----- Structures and typedefs------
typedef std::complex<double> dcomp;

// Variables for actuation.
struct ControlU{
    double tx, ty;
    int ho_sign;
    int ho_ix;
    Eigen::Matrix<double, N_ACTUATORS, 1> DM;
    Eigen::Matrix<double, 2,2> R; //Rotation matrix.
};

// This is our knowledge of the DM modes
struct ControlA{
    Eigen::Matrix<double, N_MODES, 1> modes;
    Eigen::Matrix<double, N_ACTUATORS, N_MODES> influence_functions;
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

//-------End of Commander structs------

// Settings including a mutex.
struct PIDSettings{
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
extern int beam, width, sz;

// Servo parameters. These are the parameters that will be adjusted by the commander
extern int servo_mode;
extern PIDSettings settings;
extern RTStatus rt_status;
extern ControlU control_u;
extern ControlA control_a;
extern long unsigned int cnt;

// Images - plus, minus and average
extern double *im_av, *im_plus, *im_minus;
extern float *im_plus_sum, *im_minus_sum;
extern std::mutex im_mutex;
extern TTMet_save ttmet_save;

// Main thread function for fringe tracking.
void servo_loop();



