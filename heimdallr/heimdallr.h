#include <complex> 
#include <stdarg.h>
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
#include <semaphore.h>
#include <sstream>
// This shouldn't be here, but I put the logging stuff in commander.
#include <commander/commander.h>

//----------Defines-----------
//#define SIMULATE
#define OPD_PER_DM_UNIT 6.0 
#define OPD_PER_PIEZO_UNIT 0.15 //Should be 0.26 

#define FT_STARTING 0
#define FT_RUNNING 1
#define FT_STOPPING 2

// Fast servo type
#define SERVO_FIGHT 0
#define SERVO_LACOUR 1
#define SERVO_STOP 2
#define SERVO_SIMPLE 3
#define SERVO_OFF 4

// Slow (offload) servo type
#define OFFLOAD_NESTED 0
#define OFFLOAD_GD 1
#define OFFLOAD_OFF 2
#define OFFLOAD_MANUAL 3
#define OFFLOAD_MOD 4

// The maximum number of frames to average for group delay. Delay error in wavelength from group
// delay can be 0.4/which scales to a phasor error of 0.04, while phase error can only be 0.2
// Group delay has naturally an SNR that is 2.5 times lower, so the SNR ratio is 0.2/0.04*2.5 = 12.5
// ... this means we need 12.5^2 = ~150 times more frames to average for group delay than for
// phase delay.
#define INIT_N_GD_BOXCAR 128
#define MAX_N_GD_BOXCAR 512 
#define MAX_N_PD_BOXCAR 256  // Maximum number of frames to keep for phase delay history (phasor and phase)
#define MAX_N_BS_BOXCAR 256   // Maximum number of frames to average for bispectrum
///!!! Always set to the maximum, and not synced to GD_BOXCAR.
#define MAX_N_PS_BOXCAR 256   // Maximum number of frames to average for power spectrum

#define N_TEL 4 // Number of telescopes
#define N_BL 6  // Number of baselines
#define N_CP 4  // Number of closure phases

#define DELAY_MOVE_USEC 200000 // Time to wait for the delay line to move

//----------Constant Arrays-----------
const short beam2baseline[N_TEL][N_TEL] = {
    {-1, 0, 1, 2},
    {0, -1, 3, 4},
    {1, 3, -1, 5},
    {2, 4, 5, -1}
};

const short baseline2beam[N_BL][2] = {
    {0, 1},
    {0, 2},
    {0, 3},
    {1, 2},
    {1, 3},
    {2, 3}
};

const short beam_baselines[N_TEL][N_TEL-1] = {
    {0, 1, 2},
    {0, 3, 4},
    {1, 3, 5},
    {2, 4, 5}
};

const double M_pseudo_inverse[N_TEL][N_BL] = {
    {-0.25, -0.25, -0.25, 0, 0, 0},
    {0.25, 0, 0, -0.25, -0.25, 0},
    {0, 0.25, 0, 0.25, 0, -0.25},
    {0, 0, 0.25, 0, 0.25, 0.25}
};

const short closure2bl[N_CP][3] = {
    {0, 3, 1},
    {0, 4, 2},
    {1, 5, 2},
    {3, 5, 4}
};

// These are from Lacour et al. 2019, so we'll keep the same definitions. However,
// The matrix should be an Eigen matrix for rapid multiplication later.
const Eigen::Matrix<double, N_BL, N_TEL> M_lacour = (Eigen::Matrix<double, N_BL, N_TEL>() << 
    -1, 1, 0, 0,
    -1, 0, 1, 0,
    -1, 0, 0, 1,
    0, -1, 1, 0,
    0, -1, 0, 1,
    0, 0, -1, 1).finished();
// Also, we need the pseudo-inverse of this matrix for the inverse operation.
const Eigen::Matrix<double, N_TEL, N_BL> M_lacour_dag = (Eigen::Matrix<double, N_TEL, N_BL>() << 
   -0.25,-0.25, -0.25,    0,    0,    0,
    0.25,    0,     0,-0.25,-0.25,    0,
    0,    0.25,     0, 0.25,    0,-0.25,
    0,       0,  0.25,    0, 0.25, 0.25).finished();

/* const char M[N_BL][N_TEL] = {
    {-1, 1, 0, 0},
    {-1, 0, 1, 0},
    {-1, 0, 0, 1},
    {0, -1, 1, 0},
    {0, -1, 0, 1},
    {0, 0, -1, 1}
};*/

//----- Structures and typedefs------
typedef std::complex<double> dcomp;

// As we are using Eigen and not C, we will package data from many baselines into
// a single struct
struct ControlU{
    Eigen::Vector4d dl;
    Eigen::Vector4d piezo;
    Eigen::Vector4d dm_piston;
    Eigen::Vector4d search;
    Eigen::Vector4d dl_offload;
    Eigen::Vector4d beams_active_vec = Eigen::Vector4d::Ones();
    double search_delta, dit, nbreads, tsig_len;
    unsigned int search_Nsteps, steps_to_turnaround;
    int test_beam, test_n, test_ix;
    double test_value;
    bool fringe_found;
    double itime;
};

// This is our knowledge of the per-telescope delay state. Units are all in K1 wavelengths.
struct ControlA{
    Eigen::Vector4d gd, pd, pd_phasor_boxcar_avg;
};

struct Baselines{
    Eigen::Matrix<double, N_BL, 1> gd, pd, gd_snr, pd_snr, v2_K1, v2_K2, pd_av_filtered, pd_av;
    Eigen::Matrix<dcomp, N_BL, 1> gd_phasor, pd_phasor, gd_phasor_offset;
    Eigen::Matrix<dcomp, N_BL, 1> gd_phasor_boxcar[MAX_N_GD_BOXCAR];
    Eigen::Matrix<dcomp, N_BL, 1> pd_phasor_boxcar_avg;
    Eigen::Matrix<dcomp, N_BL, 1> pd_phasor_boxcar[MAX_N_PD_BOXCAR];
    unsigned int n_gd_boxcar, n_pd_boxcar;
    // Set n_gd_boxcar and zero gd_phasor_boxcar
    void set_gd_boxcar(unsigned int n) {
        n_gd_boxcar = n;
        for (unsigned int i = 0; i < MAX_N_GD_BOXCAR; i++) {
            gd_phasor_boxcar[i].setZero();
        }
        gd_phasor.setZero();
    }
};

struct Bispectrum{
    dcomp bs_phasors[MAX_N_BS_BOXCAR];
    dcomp bs_phasor;
    double closure_phase;
    int n_bs_boxcar, ix_bs_boxcar;
};

struct FourierSampling{
    double x_px_K1[N_BL], y_px_K1[N_BL], x_px_K2[N_BL], 
        y_px_K2[N_BL], sign[N_BL];
};

/* !!! For another day - harder to serialise...
struct EncodedBaselineImages
{
    std::vector<EncodedImage> K1;
    std::vector<EncodedImage> K2;
};
*/

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
    std::vector<double> gd_bl, pd_bl;
    std::vector<double> gd_tel, pd_tel;
    std::vector<double> gd_snr, pd_snr;
    std::vector<double> closure_phase_K1, closure_phase_K2;
    std::vector<double> v2_K1, v2_K2;
    std::vector<double> dl_offload, dm_piston;
    std::vector<double> pd_av, pd_av_filtered;
    std::vector<double> gd_phasor_real, gd_phasor_imag;
    int test_ix, test_n;
    unsigned int cnt, num_ft_frames_missed;
    bool locked;
    double itime;
};

// Settings struct for commander
struct Settings
{
    unsigned int n_gd_boxcar;
    double gd_threshold;
    double pd_threshold;
    double gd_search_reset;
    unsigned int offload_time_ms;
    double offload_gd_gain;
    double gd_gain;
    double kp;
    double search_delta;
    double target_itime;
    std::string delay_line_type;
    int offload_mode, servo_mode, fixed_dl, loglevel;
    std::vector<double> search_offset;
};

//-------End of Commander structs------

struct LocalSettings {
    Settings s;
    std::mutex mutex;
};

// -------- Extern global definitions ------------
extern IMAGE DMs[N_TEL];
extern IMAGE master_DMs[N_TEL];
// The statit initial input parameters
extern toml::table config;

// Servo parameters. These are the parameters that will be adjusted by the commander
extern LocalSettings settings;
extern ControlU control_u;
extern ControlA control_a;
extern Baselines baselines;
extern FourierSampling fs;
extern Bispectrum bispectra_K1[N_CP];
extern Bispectrum bispectra_K2[N_CP];
extern double gd_to_K1;
extern long unsigned int ft_cnt;
extern int mod_ix;
extern long unsigned int nerrors;
extern bool foreground_in_place;

// Generally, we either work with beams or baselines, so have a separate lock for each.
extern std::mutex baseline_mutex, beam_mutex;
extern std::atomic<bool> zero_offload;
// DL offload variables
extern bool keep_offloading;
extern Eigen::Vector4d search_offset;
extern Eigen::Vector4d next_offload, mod_offload;

// ForwardFt class
class ForwardFt {   
public:
    // We need a mutex in case we want to change parameters while the thread is running
    // We also need a mutex for writing to the FT used for the reverse_ft
    std::mutex mutex, reverse_ft_mutex, baseline_power_mutex;
    // POSIX semaphore for new frame notification, and
    // for reverse FT ready.
    sem_t sem_new_frame;
    sem_t sem_reverse_ft_ready;
    // Public just so we set priorities in one place.
    std::thread thread, reverse_thread; 

    // Count of the frame number that has been processed
    long unsigned int cnt=0;
    short filternum; // Are we K1 or K1?
    
    // Count of the number of errors
    int nerrors=0;

    // The Fourier transformed image.
    fftw_complex *ft, *ft_copy;

    // The boxcar averaged baseline power.
    float *baseline_power_boxcar[N_BL][MAX_N_GD_BOXCAR];
    float *baseline_power_avg[N_BL];

    // Is a frame bad? This needs to be a flag so that we can 
    // monitor skipped frames in the fringe tracker.
    bool bad_frame=false;

    // A vector of bad pixel x indices
    std::vector<unsigned int> bad_pixel_x;
    std::vector<unsigned int> bad_pixel_y;

    /// The power spectrum of the image, and the array to boxcar average.
    double *power_spectra[MAX_N_PS_BOXCAR];
    double *power_spectrum;
    double *subim;
    double power_spectrum_bias;
    double power_spectrum_inst_bias;
    int ps_index = MAX_N_PS_BOXCAR-1;

    // The size of the subimage, needed to determine which Fourier components to use.
    unsigned int subim_sz, rft_sz;

    // The image that contains the metadata.
    IMAGE *subarray;

    // Constructur - just needs an IMAGE to work on
    ForwardFt(IMAGE * subarray_in);
    
    // Spawn the thread that does the FFTs.
    void start();

    // Clean-up and join the FFT thread.
    void stop();
    
    void set_bad_pixels(std::vector<unsigned int> kx, std::vector<unsigned int> ky);
private:
    // The window function to apply to the image before FFT.
    double *window;
    fftw_complex *ift_result, *ift;
    fftw_plan plan, rplan;
    std::atomic<int> mode{FT_STARTING};
    void loop();
    void reverse_ft();
};

//The forward Fourier transforms
extern ForwardFt *K1ft, *K2ft;

template <typename T>
inline std::string log_stringify(const T& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

// fringe_tracker.cpp functions
// Main thread function for fringe tracking.
void start_modulation();
void end_modulation();
void fringe_tracker();

// cam_client.cpp functions
// Camera status polling client
void start_camera_client();
void stop_camera_client();


// dl_offload.cpp functions
// Seeting the delay lines (needed form the main thread and from the commander)
void set_delay_lines(Eigen::Vector4d dl);

// Delay line offloads
extern sem_t sem_offload;
bool initialize_delay_line(std::string type);
void set_delay_lines(Eigen::Vector4d dl);
void set_mod(Eigen::Vector4d dl);
void add_to_delay_lines(Eigen::Vector4d dl);
void set_delay_line(int dl, double value);
void dl_offload();
void start_search(uint search_dl_in, double start, double stop, double rate, uint dt_ms, double threshold);

