#include <ImageStreamIO.h>
#include <stdlib.h>
#include <fitsio.h>
#include <iostream>
#define TOML_HEADER_ONLY 0
#include <toml.hpp>
#include <mutex>
#include <thread>
#include <boost/circular_buffer.hpp>
#include <condition_variable>
#include <atomic>
//#include <string>
//#include <toml++/toml.h>
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <stdexcept>

//20-8-25
#include <memory>
#include "controllers/controller.h"


#include "burst_window.h"

#pragma once
void reset_signal_injection_runtime(); // to allow baldr.cpp to have generic reset function.


//#include <ImageStreamIO/ImageStreamIO.h>

namespace dm {

// Read-only list of command-space modes (each is length = DM actuator count)
using ModeList = std::vector<Eigen::VectorXd>;

// Ensure a basis FITS is loaded if none is loaded yet (thread-safe).
bool ensure_loaded_from(const std::string& fits_path);

// Convenience: ensure using the default FITS path compiled in baldr.cpp.
bool ensure_loaded_default();

// Lookup a loaded basis by (case-insensitive) key. Returns nullptr if missing.
std::shared_ptr<const ModeList> get_basis(const std::string& basis_key_lower);

} // namespace dm


//----------Defines-----------

#define N_TEL 4 // Number of telescopes
#define N_BL 6  // Number of baselines
#define N_CP 4  // Number of closure phases

// Different types of servo modes.
#define SERVO_PID 0 // PID servo mode
#define SERVO_STOP -1
#define SERVO_CLOSE 1
#define SERVO_OPEN 0

// add this prototype BEFORE bdr_rtc_config definition
std::unique_ptr<Controller>
make_controller(const std::string& type, const bdr_controller& cfg);




//declare the helper so others can call it
void mark_injection_changed();

//----------Constant Arrays-----------

//----- Structures and typedefs------
// Declare ZMQ interface functions
void init_cam_zmq();
std::string send_cam_cmd(const std::string& command);
std::string extract_value(const std::string& response);
float get_float_cam_param(const std::string& command);


// just putting everything here for now - will move to commander when blantely obvious that I need to


//-------Commander structs-------------



// Converts a TOML array (2D array of numbers) to an Eigen matrix.
//Eigen::MatrixXd convertTomlArrayToEigenMatrix(const toml::array& arr);
//convertTomlArrayToEigenMatrix(const toml::array& arr, const Derived& /*dummy*/)
// Converts a TOML array (2D array of booleans) to an Eigen boolean matrix.
//Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> convertTomlArrayToBoolMatrix(const toml::array& arr);


// Template conversion function that takes a TOML array and a dummy
// variable of the desired Eigen type. The dummy parameter is only used
// to deduce the return type. This is useful because we pre-define our struct types and just need to 
// populate them correctly. 
// Also added extra field name at the end which checks size of input (toml) vector/matrix 
// against what is expected according to the struct. If inconsistent it fails and tells you
// precisely where and why (what the size mismatch is)
template<typename Derived>
Derived convertTomlArrayToEigenMatrix(const toml::array& arr, const Derived& /*dummy*/, const std::string& fieldName = "") {
    using Scalar = typename Derived::Scalar;
    size_t rows = arr.size();
    size_t cols = 1;
    if (rows > 0 && arr[0].as_array()) {
        cols = arr[0].as_array()->size();
    }
    
    if(Derived::RowsAtCompileTime != Eigen::Dynamic && Derived::RowsAtCompileTime != static_cast<int>(rows)) {
        std::ostringstream oss;
        oss << "Field \"" << fieldName << "\": Expected " << Derived::RowsAtCompileTime 
            << " rows but got " << rows;
        throw std::runtime_error(oss.str());
    }
    if(Derived::ColsAtCompileTime != Eigen::Dynamic && Derived::ColsAtCompileTime != static_cast<int>(cols)) {
        std::ostringstream oss;
        oss << "Field \"" << fieldName << "\": Expected " << Derived::ColsAtCompileTime 
            << " cols but got " << cols;
        throw std::runtime_error(oss.str());
    }
    
    Derived result;
    result.resize(rows, cols);
    if (cols == 1) {
        for (size_t i = 0; i < rows; ++i) {
            auto val_opt = arr[i].value<double>();
            result(i, 0) = val_opt ? static_cast<Scalar>(*val_opt) : Scalar(0);
        }
    } else {
        for (size_t i = 0; i < rows; ++i) {
            const auto* rowArr = arr[i].as_array();
            if (rowArr) {
                for (size_t j = 0; j < cols; ++j) {
                    auto val_opt = rowArr->at(j).value<double>();
                    result(i, j) = val_opt ? static_cast<Scalar>(*val_opt) : Scalar(0);
                }
            }
        }
    }
    return result;
}


inline void write_eigen_matrix_to_fits(const Eigen::MatrixXd& mat, const std::string& filename, const std::string& extname) {
    fitsfile* fptr;
    int status = 0;

    long naxes[2] = { static_cast<long>(mat.cols()), static_cast<long>(mat.rows()) };

    // Create a new FITS file (overwrite if exists)
    fits_create_file(&fptr, ("!" + filename).c_str(), &status);
    if (status) throw std::runtime_error("FITS: Could not create file");

    // Create primary image HDU
    fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status);
    if (status) throw std::runtime_error("FITS: Could not create primary image");

    // Write data: Eigen is row-major but FITS expects column-major
    Eigen::MatrixXd col_major = mat.transpose(); // transpose to column-major
    fits_write_img(fptr, TDOUBLE, 1, naxes[0]*naxes[1], col_major.data(), &status);
    if (status) throw std::runtime_error("FITS: Could not write image data");

    // Rename primary HDU
    fits_update_key(fptr, TSTRING, "EXTNAME", const_cast<char*>(extname.c_str()), nullptr, &status);

    fits_close_file(fptr, &status);
    if (status) throw std::runtime_error("FITS: Error closing file");
}

inline void updateDMSharedMemory(IMAGE &dmImg, const Eigen::VectorXd &dmCmd) {
    auto *md = dmImg.md;
    int N = md->nelement;

    // Entry log
    //std::cout << "[DBG] updateDMSharedMemory() called with dmCmd.size() = "
    //          << dmCmd.size() << " (expecting " << N << ")\n";

    // Check size
    if ((int)dmCmd.size() != N) {
        std::cerr << "[ERROR] size mismatch: got " << dmCmd.size()
                  << " vs " << N << "\n";
        return;
    }

    // Log a few sample values from the command
    //std::cout << "[DBG] dmCmd first 5 vals: ";
    //for (int i = 0; i < std::min(5, N); ++i) std::cout << dmCmd[i] << " ";
    //std::cout << "\n";

    // Print metadata before write
    //std::cout << "[DBG] Before write: cnt0=" << md->cnt0
    //          << "  cnt1=" << md->cnt1
    //          << "  write=" << md->write << "\n";

    // Mark write
    md->write = 1;
    std::atomic_thread_fence(std::memory_order_release);

    // Copy the data
    std::memcpy(dmImg.array.D, dmCmd.data(), N * sizeof(double));

    // Bump counters
    md->cnt0++;
    md->cnt1++; //!!! MJI, not sure why this was set to 0.

    // Post semaphore
    int ret = ImageStreamIO_sempost(&dmImg, -1);
    //std::cout << "[DBG] ImageStreamIO_sempost returned " << ret << "\n";

    // Clear write flag
    md->write = 0;


}

//------------------------------------------------------------------------------
// Drain any outstanding semaphore “posts” so that
// the next semwait() really waits for a fresh frame.
//------------------------------------------------------------------------------
static inline void catch_up_with_sem(IMAGE* img, int semid) {
    // keep grabbing until there are no more pending posts
    while (ImageStreamIO_semtrywait(img, semid) == 0) { /* nothing just do it*/; }
}


// Declaration of readConfig function
toml::table readConfig(const std::string &filename);

//bdr_rtc_config readBDRConfig(const toml::table& config, const std::string& beamKey, const std::string& phaseKey);

//-----------------------------------------------------
// bdr_method: top-level method definitions.
struct bdr_state {
    std::string DM_flat;          // e.g., "baldr"
    std::string signal_space;   // where we consider our signals (pixel or dm)
    int LO;                       // low-order modes (e.g., 2)
    std::string controller_type;  // e.g., "PID"
    std::string inverse_method_LO;   // e.g., "map"
    std::string inverse_method_HO;   // e.g., "map"
    std::string phasemask; // what phasemask are we using 
    int auto_close; // do we close loops automatically? 
    int auto_open;// do we open loops automatically?  
    int auto_tune; // do we automatically tune gains?
    int take_telemetry; // do we automatically tune gains?
    int simulation_mode ;// simulatie camera and DM 
    // Validate method.
    void validate() const {
        if (DM_flat.empty())
            throw std::runtime_error("bdr_state: DM_flat is empty.");
        if (LO <= 0)
            throw std::runtime_error("bdr_state: LO must be positive.");
        if (controller_type.empty())
            throw std::runtime_error("bdr_state: controller_type is empty.");
        if (inverse_method_LO.empty())
           throw std::runtime_error("bdr_state: inverse_method_LO is empty.");
        if (inverse_method_HO.empty())
           throw std::runtime_error("bdr_state: inverse_method_HO is empty.");
    }
};

//-----------------------------------------------------
// bdr_reduction: Beam-specific reduction products.
struct bdr_reduction {
    Eigen::VectorXd bias;      // Mean (flattened) bias frame (ADU)
    Eigen::VectorXd bias_dm;   // Bias intensity interpolated to DM pixels (ADU)
    Eigen::VectorXd dark;      // Mean (flattened) dark frame (ADU/s/gain)
    Eigen::VectorXd dark_dm;   // Dark intensity interpolated to DM pixels (ADU/s/gain)

    void project_to_dm( Eigen::MatrixXd I2A ){
        bias_dm = I2A * bias; 
        dark_dm = I2A * dark; 
    }

    void validate() const {
        if (bias.size() == 0 || bias_dm.size() == 0 || dark.size() == 0 || dark_dm.size() == 0)
            throw std::runtime_error("bdr_reduction: One or more vectors are empty.");
        // You might want to check that bias_dm has the same size as bias, etc.
    }
};

//-----------------------------------------------------
// bdr_pixels: Pixel indices.
struct bdr_pixels {
    Eigen::Matrix<int16_t, Eigen::Dynamic, 1> crop_pixels; // r1, r2, c1, c2 for cropping.
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1> bad_pixels;      // Indices (flattened)
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1> pupil_pixels;      // Pupil pixel indices.
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1> interior_pixels;   // Interior pupil indices.
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1> secondary_pixels;  // Secondary pixel indices.
    Eigen::Matrix<int32_t, Eigen::Dynamic, 1> exterior_pixels;   // Exterior pixel indices.

    void validate() const {
        // You can add size checks or consistency tests.
        if (crop_pixels.size() != 4)
            throw std::runtime_error("bdr_pixels: crop_pixels must have 4 elements.");
    }
};

//-----------------------------------------------------
// bdr_refence_pupils: Reference intensities.
struct bdr_refence_pupils {
    Eigen::VectorXd I0;         // Reduced reference intensity unitless //(ADU/s/gain) [flattened]
    Eigen::VectorXd N0;         // Reduced reference intensity unitless //(ADU/s/gain)
    Eigen::VectorXd norm_pupil; // Filtered reference intensity unitless //(ADU/s/gain)
    Eigen::VectorXd norm_pupil_dm; // Interpolated to DM pixels unitless //(ADU/s/gain)
    Eigen::VectorXd I0_dm;      // Interpolated to DM pixels unitless //(ADU/s/gain)
    double intrn_flx_I0; // internal source flux in ADU - used for Strehl estimation
    
    // update all dm reference pupils
    void project_to_dm( Eigen::MatrixXd I2A ){
        norm_pupil_dm = I2A * norm_pupil; 
        I0_dm = I2A * I0; 
    }

    // to do them seperately if we want to update 
    void project_I0_to_dm( Eigen::MatrixXd I2A ){
        I0_dm = I2A * I0; 
    }

    // to do them seperately if we want to update 
    void project_N0norm_to_dm( Eigen::MatrixXd I2A ){
        norm_pupil_dm = I2A * norm_pupil; 
    }


    void validate() const {
        if (I0.size() == 0 || N0.size() == 0 || norm_pupil.size() == 0)
            throw std::runtime_error("bdr_refence_pupils: one or more vectors are empty.");
    }
};

//-----------------------------------------------------
// bdr_matricies: Matrices for processing.
struct bdr_matricies {
    Eigen::MatrixXd I2A;      // Interpolation matrix (pixels to DM)
    Eigen::MatrixXd I2M;      // Intensity-to-mode projection matrix.
    Eigen::MatrixXd I2M_LO;   // For low-order modes.
    Eigen::MatrixXd I2M_HO;   // For high-order modes.
    Eigen::MatrixXd M2C;      // Mode-to-DM command projection matrix.
    Eigen::MatrixXd M2C_LO;   // LO mode to DM command.
    Eigen::MatrixXd M2C_HO;   // HO mode to DM command.
    Eigen::MatrixXd I2rms_sec; // For secondary obstruction. - keep this in matrix form so more complex models can be extended
    Eigen::MatrixXd I2rms_ext; // For exterior pupil.- keep this in matrix form so more complex models can be extended
    // Additional size variables:
    int32_t szm; // Number of modes.
    int32_t sza; // Number of actuators.
    int32_t szp; // Number of pixels.

    void validate() const {
        // Example: check that dimensions are consistent.
        if (I2A.rows() == 0 || I2A.cols() == 0)
            throw std::runtime_error("bdr_matricies: I2A matrix is empty.");
        // Further consistency checks can be added.
    }
};

//-----------------------------------------------------
// bdr_controller: AO controller settings.
// 20-8-25
struct bdr_controller {
    Eigen::VectorXd kp;
    Eigen::VectorXd ki;
    Eigen::VectorXd kd;
    Eigen::VectorXd lower_limits;
    Eigen::VectorXd upper_limits;
    Eigen::VectorXd set_point;

    bdr_controller(int vsize = 140)
        : kp(Eigen::VectorXd::Zero(vsize)),
          ki(Eigen::VectorXd::Zero(vsize)),
          kd(Eigen::VectorXd::Zero(vsize)),
          lower_limits(Eigen::VectorXd::Constant(vsize, -2.0)),
          upper_limits(Eigen::VectorXd::Constant(vsize, +2.0)),
          set_point(Eigen::VectorXd::Zero(vsize)) {}

    void validate() const {
        const auto n = kp.size();
        if (n == 0 || ki.size() == 0 || kd.size() == 0)
            throw std::runtime_error("bdr_controller: one or more gain vectors are empty.");

        // All vectors must be same size
        if (ki.size()!=n || kd.size()!=n ||
            lower_limits.size()!=n || upper_limits.size()!=n || set_point.size()!=n)
            throw std::runtime_error("bdr_controller: vector size mismatch.");

        // Ensure lower <= upper elementwise
        for (Eigen::Index i=0;i<n;++i) {
            if (lower_limits[i] > upper_limits[i])
                throw std::runtime_error("bdr_controller: lower_limits > upper_limits at index " + std::to_string(i));
        }
    }
};

// struct bdr_controller {
//     // For now, only gains. In a complete system you might add anti-windup limits, etc.
//     Eigen::VectorXd kp;
//     Eigen::VectorXd ki;
//     Eigen::VectorXd kd;
//     Eigen::VectorXd lower_limits;
// 	Eigen::VectorXd upper_limits;
// 	Eigen::VectorXd set_point;
//     void validate() const {
//         if (kp.size() == 0 || ki.size() == 0 || kd.size() == 0)
//             throw std::runtime_error("bdr_controller: one or more gain vectors are empty.");
//         if ( (kp.size() != ki.size()) || (kp.size() != kd.size()))
//             throw std::runtime_error("bdr_controller: gain vector sizes mismatch.");
//     }
//     // Constructor with default vector size of 5.
//     bdr_controller(int vsize = 140)
//         : kp(Eigen::VectorXd::Zero(vsize)),
//           ki(Eigen::VectorXd::Zero(vsize)),
//           kd(Eigen::VectorXd::Zero(vsize)),
//           lower_limits(Eigen::VectorXd::Constant(vsize, -2)),
//           upper_limits(Eigen::VectorXd::Constant(vsize, 2)),
//           set_point(Eigen::VectorXd::Zero(vsize))
//         {}
// };


//20-8-25
// // Declaration of the PIDController class. needs to be after bdr_controller struct since it uses it in the constructor
// class PIDController {
// public:
//     // Constructors.
//     PIDController(const Eigen::VectorXd& kp_in,
//                   const Eigen::VectorXd& ki_in,
//                   const Eigen::VectorXd& kd_in,
//                   const Eigen::VectorXd& lower_limit_in,
//                   const Eigen::VectorXd& upper_limit_in,
//                   const Eigen::VectorXd& setpoint_in);
//     PIDController(const bdr_controller& config_in);
//     PIDController();

//     // Process the measured input to produce control output.
//     Eigen::VectorXd process(const Eigen::VectorXd& measured);

//     // Utility functions.
//     void set_all_gains_to_zero();
//     void reset();

//     // Public members 
//     Eigen::VectorXd kp;
//     Eigen::VectorXd ki;
//     Eigen::VectorXd kd;
//     Eigen::VectorXd lower_limits;
//     Eigen::VectorXd upper_limits;
//     Eigen::VectorXd set_point;
//     std::string ctrl_type;
//     Eigen::VectorXd output;
//     Eigen::VectorXd integrals;
//     Eigen::VectorXd prev_errors;
// };


//-----------------------------------------------------
// bdr_limits: Limits and conditions for loop control.
struct bdr_limits {
    float close_on_strehl_limit; // when should we automatically try to close the loop?
    float open_on_strehl_limit; // at what level of Strehl (estimate) do we open the loop?
    float open_on_flux_limit; // at what level of flux do we open the loop?
    float open_on_dm_limit; // at what level of DM (rms? or peaktovalley) do we open the loop (or limit gains)?
    float LO_offload_limit; // at what level do we try offload the lower order modes ?

    void validate() const {

    }
};



//-----------------------------------------------------
// bdr_cam: Camera configuration.
struct bdr_cam {
    std::string fps;
    std::string gain;
    std::string testpattern;
    std::string bias;
    std::string flat;
    std::string imagetags;
    std::string led;
    std::string events;
    std::string extsynchro;
    std::string rawimages;
    std::string cooling;
    std::string mode;
    std::string resetwidth;
    std::string nbreadworeset;
    std::string cropping;
    std::string cropping_columns;
    std::string cropping_rows;
    std::string aduoffset;
    
    void validate() const {
        // Check required fields; for now, we assume they all must be non-empty.
        if (fps.empty() || gain.empty())
            throw std::runtime_error("bdr_cam: fps and gain must be provided.");
    }
};

//-----------------------------------------------------
// // bdr_telem: Telemetry configuration.
struct bdr_telem {
    int counter;  // A counter that increments for each telemetry sample.
    
    // For timestamp, we store a single double (e.g., seconds or milliseconds since an epoch)
    boost::circular_buffer<double> timestamp; 
    boost::circular_buffer<int> LO_servo_mode; // this is useful for state change conditions in RTC 
    boost::circular_buffer<int> HO_servo_mode;
    boost::circular_buffer<Eigen::VectorXd> img;   
    boost::circular_buffer<Eigen::VectorXd> img_dm;       
    boost::circular_buffer<Eigen::VectorXd> signal;    
    boost::circular_buffer<Eigen::VectorXd> e_LO;
    boost::circular_buffer<Eigen::VectorXd> u_LO;
    boost::circular_buffer<Eigen::VectorXd> e_HO;
    boost::circular_buffer<Eigen::VectorXd> u_HO;
    boost::circular_buffer<Eigen::VectorXd> c_LO;
    boost::circular_buffer<Eigen::VectorXd> c_HO;
    boost::circular_buffer<Eigen::VectorXd> c_inj; //<-- NEW NEW 
    boost::circular_buffer<double> rmse_est;   // <-- NEW
    boost::circular_buffer<double> snr;         // <-- NEW

    // Single source of truth for "numeric telemetry fields we can save".
    static inline constexpr std::array<std::string_view, 15> kNumericSavableFields = {
        "timestamp",
        "LO_servo_mode",
        "HO_servo_mode",
        "img",
        "img_dm",
        "signal",
        "e_LO",
        "u_LO",
        "e_HO",
        "u_HO",
        "c_LO",
        "c_HO",
        "c_inj",
        "rmse_est",
        "snr"
    };
    // Constructor that sets a fixed capacity for each ring buffer.
    bdr_telem(size_t capacity = 10) // 10000 capacity is about 150 MB in the buffer
      : counter(0),
        timestamp(capacity),
        LO_servo_mode(capacity),
        HO_servo_mode(capacity),
        img(capacity),
        img_dm(capacity),
        signal(capacity),
        e_LO(capacity),
        u_LO(capacity),
        e_HO(capacity),
        u_HO(capacity),
        c_LO(capacity),
        c_HO(capacity),
        c_inj(capacity), 
        rmse_est(capacity),   // <-- initialize
        snr(capacity)         // <-- initialize
    {}
    
    // Method to update capacity for all ring buffers.
    void setCapacity(size_t newCapacity) {
        timestamp.set_capacity(newCapacity);
        LO_servo_mode.set_capacity(newCapacity);
        HO_servo_mode.set_capacity(newCapacity);
        img.set_capacity(newCapacity);
        img_dm.set_capacity(newCapacity);
        signal.set_capacity(newCapacity);
        e_LO.set_capacity(newCapacity);
        u_LO.set_capacity(newCapacity);
        e_HO.set_capacity(newCapacity);
        u_HO.set_capacity(newCapacity);
        c_LO.set_capacity(newCapacity);
        c_HO.set_capacity(newCapacity);
        c_inj.set_capacity(newCapacity); // <-- NEW NEW 
        rmse_est.set_capacity(newCapacity);  // <-- new
        snr.set_capacity(newCapacity); // <-- new
    }
    //void validate() const {
    // Add any validation if needed.
    //}
};

//-----------------------------------------------------
// bdr_filters: Boolean masks.
struct bdr_filters {
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> bad_pixel_mask;
    Eigen::VectorXf bad_pixel_mask_dm;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> pupil;
    Eigen::VectorXf pupil_dm;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> secondary;
    Eigen::VectorXf secondary_dm;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> exterior;
    Eigen::VectorXf exterior_dm;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 1> inner_pupil_filt;
    Eigen::VectorXf inner_pupil_filt_dm;

    void validate() const {
        }

    // Project all masks to DM space.
    void project_to_dm(const Eigen::MatrixXd& I2A) {
        std::cout << "I2A.cols() = " << I2A.cols() << std::endl;
        std::cout << "bad_pixel_mask.size() = " <<  bad_pixel_mask.size() << std::endl;
        // Do similar prints for pupil, secondary, etc.
        // Assume 'mask' is one of your Eigen vector masks, e.g., exterior.
        Eigen::VectorXd mask_d = exterior.cast<double>();

        // Debug prints: Print I2A number of columns and the mask size.
        std::cout << "I2A dimensions: " << I2A.rows() << " x " << I2A.cols() << std::endl;
        std::cout << "Mask (exterior) size: " << mask_d.size() << std::endl;

        // Convert each mask to a double vector, multiply by I2A, then cast to float.
        bad_pixel_mask_dm = (I2A * bad_pixel_mask.cast<double>()).cast<float>();
        pupil_dm = (I2A * pupil.cast<double>()).cast<float>();
        secondary_dm = (I2A * secondary.cast<double>()).cast<float>();
        exterior_dm = (I2A * exterior.cast<double>()).cast<float>();
        inner_pupil_filt_dm = (I2A * inner_pupil_filt.cast<double>()).cast<float>();
    }

};

// Signal injection (command-space) ---
struct bdr_signal_cfg {
    bool        enabled        = false;     // master on/off
    //std::string space          = "command";   // we replace this with apply_to // fixed: we inject via dm -> dm command space
    std::string basis          = "zonal";   // name used by dm::get_basis(...)
    int         basis_index    = 0;         // which mode to inject
    double      amplitude      = 0.05;      // scalar gain applied to unit-normalized mode
    std::string waveform       = "sine";    // "sine","square","step","chirp","prbs","none"
    double      freq_hz        = 10.0;      // for sine/square
    double      phase_deg      = 0.0;       // phase offset
    double      duty           = 0.5;       // for square (0..1)
    double      t_start_s      = 0.0;       // start time gate
    double      t_stop_s       = 0.0;       // 0 => no stop
    int         latency_frames = 0;         // simulate pipeline delay
    int         hold_frames    = 1;         // sample-and-hold; 1 => update every frame
    uint32_t    prbs_seed      = 0xACE1u;   // initial seed (PRBS)
    // Chirp options:
    double      chirp_f0       = 1.0;
    double      chirp_f1       = 50.0;
    double      chirp_T        = 5.0;       // seconds of sweep window

    std::string branch         = "HO";        // "LO" | "HO" | "ALL" <- what control branch to apply the injected signals
    std::string apply_to       = "command";   // "command" | "setpoint" <- what signal space to inject them intop
};

//-----------------------------------------------------
// Master configuration struct for RTC.
struct bdr_rtc_config {
    bdr_state state;
    bdr_reduction reduction;
    bdr_pixels pixels;
    bdr_refence_pupils reference_pupils;
    bdr_matricies matrices;
    bdr_controller ctrl_LO_config;
    bdr_controller ctrl_HO_config;
    bdr_limits limits;
    bdr_cam cam;
    bdr_telem telem;
    bdr_filters filters;
    bdr_signal_cfg inj_signal;   // <--- NEW

    // // Derived run time parameters (need to be re-calculated if we reload a config)
    //PIDController ctrl_LO ;
    //PIDController ctrl_HO ;
    // NEW 20-8-25:
    std::unique_ptr<Controller> ctrl_LO;
    std::unique_ptr<Controller> ctrl_HO;

    // some pre inits
    //Eigen::VectorXd img ; // image from SHM (size could change)
    Eigen::VectorXd zeroCmd ; // zero dm command for rtc channel

    // ----------------------- IMPORTANT 
    // frames per second from rtc_config used for converting dark from adu/s -> adu 
    // we should actually read this from the current camera settings when calling the rtc! 
    double fps ;
    double gain ;
    double scale ; // ratio of fps and gain

    //burst update here
    BurstWindow burst; // set window to hold rolling intensities in a single burst (slope estimation etc)

    // ALL I2M and reference intensities stored in config are in ADU/second/gain !!
    Eigen::MatrixXd I2M_LO_runtime ; // LO intensity to mode calibrated for fps and gain 
    Eigen::MatrixXd I2M_HO_runtime ; // HO intensity to mode calibrated for fps and gain 
    
    Eigen::VectorXd N0_dm_runtime ; // clear pupil calibrated for fps and gain 
    Eigen::VectorXd I0_dm_runtime ; // ZWFS pupil calibrated for fps and gain

    // darks (generated in dcs/calibration_frames/gen_dark_bias_badpix.py ) are adu/s in the gain setting (not normalized by gain)
    // I should review the darks and perhaps normalize by gain setting too! 
    //Eigen::VectorXd dark_dm_runtime ;

    // Strehl models
    // secondary pixel. Solarstein mask closer to UT size - this will be invalid on internal source 
    int sec_idx ; // .secondary_pixels defines 3x3 square around secondary - we only use the central one
    double m_s_runtime ; // slope - intensity is normalized by fps and gain in model (/home/asg/Progs/repos/asgard-alignment/calibration/build_strehl_model.py)
    double b_s_runtime ; // intercept for rms model (dm units)
    


    void initDerivedParameters() {

        // 20-8-25: Initialize controllers based on the state.
        ctrl_LO_config.validate();
        ctrl_HO_config.validate();
        ctrl_LO = make_controller(state.controller_type, ctrl_LO_config);
        ctrl_HO = make_controller(state.controller_type, ctrl_HO_config);
        // if (state.controller_type == "PID") {
        //     ctrl_LO = PIDController(ctrl_LO_config);
        //     ctrl_HO = PIDController(ctrl_HO_config);
        // } else {
        //     throw std::runtime_error("Invalid controller_type");
        // }

        zeroCmd = Eigen::VectorXd::Zero(matrices.sza);

        // Use ZMQ to query runtime camera settings
        float cal_gain = std::stof(cam.gain); // used in calibration of interaction matrix 
        float cal_fps = std::stof(cam.fps);

	std::cout << "reading camera fps and gain via camera server with zmq" << std::endl;
        gain = get_float_cam_param("gain raw");
        fps = get_float_cam_param("fps raw");

        std::cout << "[ZMQ] Using runtime gain = " << gain << ", fps = " << fps << std::endl;

	std::cout << "reading nbreadworeset to configure burst window" << std::endl;
        try {
            float nbread_val = get_float_cam_param("nbreadworeset raw");

            int nbread_int = static_cast<int>(nbread_val);
            if (nbread_int > 0 && fps > 0.0) {
                burst.configure(nbread_int, fps);
                std::cout << "[initDerivedParameters] Burst window configured: nbread=" << nbread_int << ", fps=" << fps << std::endl;
            } else {
                std::cerr << "[initDerivedParameters] Warning: invalid fps or nbread from camera (fps=" << fps << ", nbread=" << nbread_int << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[initDerivedParameters] Failed to configure burst window: " << e.what() << std::endl;
        }

        scale = gain / fps;

        // post TTonsky , we normalize reference intensities used to build these matricies by subframe total intensity 
        // yetto verifiy scale factor not needed here. 
        I2M_LO_runtime = matrices.I2M_LO; //scale * matrices.I2M_LO;
        I2M_HO_runtime = matrices.I2M_HO; //scale * matrices.I2M_HO;
        N0_dm_runtime = reference_pupils.norm_pupil_dm; //scale * reference_pupils.norm_pupil_dm;
        I0_dm_runtime = reference_pupils.I0_dm; //scale * reference_pupils.I0_dm;
        //dark_dm_runtime = (1.0 / fps) * (gain/cal_gain) * reduction.dark_dm; // our dark is in ADU/s but not normalized by gain, so we need to multiply by ratio of the IM gain and gain for runnning rtc. 
        // ^^ this feature should be optimized .. by considering gain normalized darks or keep input image in adu/s/gain

        sec_idx = pixels.secondary_pixels(4);
        m_s_runtime = matrices.I2rms_sec(0, 0);
        b_s_runtime = matrices.I2rms_sec(1, 1);


    }


    // before zmq camera fps and gain read in . 
    // // Function to initialize all derived runtime parameters.
    // void initDerivedParameters() {
    //     // Initialize controllers
    //     if (state.controller_type == "PID") {
    //         ctrl_LO = PIDController(ctrl_LO_config);
    //         ctrl_HO = PIDController(ctrl_HO_config);
    //     } else {
    //         std::cout << "no ctrl match" << std::endl;
    //         throw std::runtime_error("no valid controller_type in state");
    //     }
        
    //     // Initialize image and command vectors.
    //     //img = Eigen::VectorXd::Zero(matrices.szp);
    //     zeroCmd = Eigen::VectorXd::Zero(matrices.sza);
        
    //     //// -------------------- TO ADD IN 
    //     // FPS and gain handling
    //     // float cal_gain = std::stof(cam.gain);
    //     // float cal_fps = std::stof(cam.fps);

    //     // if (override_gain_fps) {
    //     //     fps = fps_override;
    //     //     gain = gain_override;
    //     //     std::cout << "[INFO] Overriding FPS = " << fps << ", Gain = " << gain << std::endl;
    //     // } else {
    //     //     fps = cal_fps;
    //     //     gain = cal_gain;
    //     //     std::cout << "[INFO] Using FPS and Gain from TOML: " << fps << ", " << gain << std::endl;
    //     // }

    //     // scale = gain / fps; // used for scaling I2M and references

    //     // // Scale interaction matrices and reference intensities
    //     // I2M_LO_runtime = scale * matrices.I2M_LO;
    //     // I2M_HO_runtime = scale * matrices.I2M_HO;
    //     // N0_dm_runtime = scale * reference_pupils.norm_pupil_dm;
    //     // I0_dm_runtime = scale * reference_pupils.I0_dm;

    //     // // Adjust dark frame using calibration gain vs runtime gain
    //     // dark_dm_runtime = (1.0 / fps) * (cal_gain / gain) * reduction.dark_dm;

    //     ///// ALSO 
    //     // in baldr.cpp before main define!
    //     // float gain_override = 1.0f;
    //     // float fps_override = 1000.0f;
    //     // bool override_gain_fps = false;

    //     /// add to end of bldr.h
    //     // extern float gain_override;
    //     // extern float fps_override;
    //     // extern bool override_gain_fps;


    //     //// -------------------- 

    //     // Convert the camera parameters (stored as strings) to doubles.
    //     fps = std::stof(cam.fps);
    //     gain = std::stof(cam.gain);
    //     scale = gain / fps;
        
    //     // Scale matrices and reference intensities.
    //     I2M_LO_runtime = scale * matrices.I2M_LO;
    //     I2M_HO_runtime = scale * matrices.I2M_HO;
    //     N0_dm_runtime = scale * reference_pupils.norm_pupil_dm;
    //     I0_dm_runtime = scale * reference_pupils.I0_dm;
        
    //     // Normalize dark according to fps.
    //     dark_dm_runtime = (1.0 / fps) * reduction.dark_dm;
    //     //dark_dm_runtime = 1 / float(IM_cam_config["gain"]) *  scale * reduction.dark_dm;
        
    //     // Strehl model parameters.
    //     // secondary_pixels is an Eigen column vector; use (4) to get the fifth element.
    //     sec_idx = pixels.secondary_pixels(4);
    //     m_s_runtime = scale * matrices.I2rms_sec(0, 0);
    //     b_s_runtime = matrices.I2rms_sec(1, 1);
    // }

    void validate() const {
        state.validate();
        reduction.validate();
        pixels.validate();
        reference_pupils.validate();
        matrices.validate();
        //heree
        //controller.validate();
        ctrl_LO_config.validate();
        ctrl_HO_config.validate();
        limits.validate();
        cam.validate();
        //telem.validate();
        filters.validate();
	std::cout << "done validating rtc.." << std::endl;
        
    }
};



struct Status {
    int TT_state = 0;
    int HO_state = 0;
    std::string mode = "";
    std::string phasemask = "";
    double frequency = 0.0;   // double
    int configured   = 1;
    std::string ctrl_type = "PID";
    std::string config_file = "";
    int inj_enabled = 0;
    int auto_loop =  1;
    double close_on_snr = 0.0; // should be a float
    double open_on_snr = 0.0; // should be a float
    double close_on_strehl = 0.0;    
    double open_on_strehl = 0.0;
    double TT_offsets = 0.0; // 1 or 0 

};



bdr_rtc_config readBDRConfig(const toml::table& config, const std::string& beamKey, const std::string& phaseKey);



// An encoded 2D image in row-major form.
// If this is to be useful, then the heimdallr EncodedImage
// should become a library.
struct EncodedImage
{
    unsigned int szx, szy;
    std::string type;
    std::string message;
};


//20-8-25
// Make a concrete controller from type string and PID-like config.
std::unique_ptr<Controller>
make_controller(const std::string& type, const bdr_controller& cfg);


//-------End of Commander structs------



// -------- Extern global definitions ------------

// Declare ZMQ functions


// The static initial input parameters
//extern toml::table config;
//extern std::vector<toml::table> config; // initial configuration 
extern int beam_id; 
extern std::string phasemask;
extern toml::table config;
extern bdr_rtc_config rtc_config;

//loop time - frame driven so don't use unless testing 
//extern float loop_time ; //us 
//extern bool loop_time_override ;

extern double g_subframe_int; // sum of intensity in the current subframe //post TTonsky

// uncomment and build July 2025 AIV
extern std::atomic<int> global_boxcar;


//extern std::vector<bdr_rtc_config> rtc_config_list; // what the rtc will use and edit
extern std::atomic<int> servo_mode; 
extern std::atomic<int> servo_mode_LO;
extern std::atomic<int> servo_mode_HO;
// extern int servo_mode;
//extern int servo_mode_LO;
//extern int servo_mode_HO;
// extern vector::<int> telescopes;

// change signal epoch to mark changes in signal injection config
extern std::atomic<uint64_t> g_inj_cfg_epoch;

/////// new stuff for onsky interactions (19/8/25)
// --- Open-loop offsets published atomically as one struct ---
// --- Open-loop offsets published atomically as one struct ---
// --- Open-loop offsets published atomically as one struct ---
struct OLOffsets {
    Eigen::VectorXd lo;  // LO branch DM-command offset (length = DM command vector)
    Eigen::VectorXd ho;  // HO branch DM-command offset (length = DM command vector)
};

// Global handle (RTC reads via atomic_load on this pointer)
extern std::shared_ptr<const OLOffsets> g_ol_offsets;

// convenience loader (no need for a separate extern/definition)
// inline std::shared_ptr<const OLOffsets> load_openloop_offsets() {
//     return std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
// }

/////// end new stuf 

// Servo parameters. These are the parameters that will be adjusted by the commander

// We at least need a mutex for RTC parameters.
extern std::mutex rtc_mutex;
extern std::mutex telemetry_mutex;
extern std::mutex ctrl_mutex;
// extern std::mutex ctrl_LO_mutex;
// extern std::mutex ctrl_HO_mutex;
extern std::atomic<bool> pause_rtc;        // When true, RTC should pause.
extern std::mutex rtc_pause_mutex;         // Protects shared access to pause state.
extern std::condition_variable rtc_pause_cv; // Notifies RTC to resume.

//
extern std::string telemFormat;
extern std::string telem_save_path;

// The C-Red Image subarray and DM
extern IMAGE subarray;
extern IMAGE dm_rtc;
extern IMAGE dm_rtc0; // master

// Main thread function for the RTC
void rtc();

// Main thread function for the telemetry
void telemetry();
