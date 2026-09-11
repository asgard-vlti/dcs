#include "baldr.h"
#include <chrono>
#include <iostream>
#include <ImageStreamIO.h>
#include <cstdint>  // for uint16_t
#include <nlohmann/json.hpp>

#include <fstream>

// adding for signal injection. 
#include <random>
#include <deque>
#include <algorithm>   // transform, clamp
#include <cctype>      // tolower
#include <cmath>       // M_PI (or define your own PI)
#include <memory>      // shared_ptr
#include <stdexcept>   // runtime_error

using json = nlohmann::json;

// json telemetry_to_json(const bdr_telem &telemetry) {
//     json j;
//     // Convert timestamp to milliseconds since epoch.
//     // Convert the vector of steady_clock time_points to a JSON array in milliseconds.
//     json ts = json::array();
//     for (const auto &t : telemetry.timestamp) {
//         // Convert the steady_clock::time_point to milliseconds.
//         auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count();
//         ts.push_back(ms);
//     }
//     j["timestamps"] = ts;   // Use "timestamps" since it's an array.
    
//     j["img"] = telemetry.img;                // nlohmann::json converts std::vector easily.
//     j["signal"] = telemetry.signal;
//     j["e_TT"] = telemetry.e_TT;
//     j["u_TT"] = telemetry.u_TT;
//     j["e_HO"] = telemetry.e_HO;
//     j["u_HO"] = telemetry.u_HO;
//     j["dm_ch0"] = telemetry.dm_ch0;
//     j["dm_ch1"] = telemetry.dm_ch1;
//     j["dm_ch2"] = telemetry.dm_ch2;
//     j["dm_ch3"] = telemetry.dm_ch3;
//     j["dm"] = telemetry.dm;
//     j["strehl_est"] = telemetry.strehl_est;
//     j["dm_rms"] = telemetry.dm_rms;
//     return j;
// }

// void write_telemetry_to_json(const bdr_telem &telemetry, const std::string &filename) {
//     json j = telemetry_to_json(telemetry);
//     std::ofstream ofs(filename);
//     if (!ofs) {
//         std::cerr << "Could not open file " << filename << " for writing." << std::endl;
//         return;
//     }
//     ofs << j.dump(4); // pretty-print with an indent of 4 spaces.
//     ofs.close();
//     std::cout << "Telemetry written to " << filename << std::endl;
// }



long unsigned int rtc_cnt=0;
unsigned int nerrors=0;

// Add local (to the RTC) variables here

// controllers - these get initialized by the rtc.ctrl_LO_config and rtc.ctrl_HO_config struct
// PIDController ctrl_LO ; 
// PIDController ctrl_HO ; 

// intermediate products 
Eigen::VectorXd img;

Eigen::VectorXd img_dm;

Eigen::VectorXd sig;

Eigen::VectorXd e_HO;
Eigen::VectorXd u_HO;

Eigen::VectorXd e_LO;
Eigen::VectorXd u_LO;

Eigen::VectorXd c_LO;
Eigen::VectorXd c_HO;
Eigen::VectorXd c_inj;
Eigen::VectorXd dmCmd;

Eigen::VectorXd ol_lo; 
Eigen::VectorXd ol_ho; 

std::vector<Eigen::VectorXd> SS;  // size M, oldest at S[0], newest at S[M-1]


//// getting telemetry in AIV 
// IMAGE shm_sig, shm_eLO, shm_eHO;
// constexpr int shm_telem_samples = 20;  // Number of telemetry samples exported to SHM
// thread_local static std::size_t shm_telem_cnt  = 0; // for counting our set window to write telem to shm. 


// dm rms model
double dm_rms_est_1 ; // from secondary obstruction
double dm_rms_est_2 ; // from exterior pixels 

// delete cause we define later from baldr boxcar global variable
//const size_t boxcar = 5; // how many samples we average



// Timing helper
using Clock = std::chrono::high_resolution_clock;


template<typename Buffer>
Eigen::VectorXd weightedAverage(const Buffer &buf, size_t K) {
    size_t M = buf.size();
    if (M == 0) 
        throw std::runtime_error("empty telemetry buffer");

    // Only consider up to the last K samples:
    size_t window = std::min(M, K);
    if (window == 1) 
        return buf[M-1];

    // We'll average buf[M-window] ... buf[M-1]
    int N = buf[0].size();
    Eigen::VectorXd acc = Eigen::VectorXd::Zero(N);
    double sumW = 0.0;

    // weights w_j = j/(window-1), j=0..window-1
    for (size_t j = 0; j < window; ++j) {
        double w = double(j) / double(window - 1);
        acc += w * buf[M - window + j];
        sumW += w;
    }

    return acc / sumW;
}




/**
 Reads a frame from shared memory which should match the expected number of pixels expectedPixels.
**/
Eigen::VectorXd getFrameFromSharedMemory(int expectedPixels) {
    if (!subarray.md) {
        //ImageStreamIO_destroyIm(&subarray);
        throw std::runtime_error("No metadata found in shared memory image.");
    }
    
    int nx = subarray.md->size[0];
    int ny = (subarray.md->naxis > 1) ? subarray.md->size[1] : 1;
    int totalPixels = nx * ny;
    
    if (totalPixels != expectedPixels) {
        //ImageStreamIO_destroyIm(&subarray);
        throw std::runtime_error("Frame size mismatch: expected " +
                                 std::to_string(expectedPixels) +
                                 ", got " + std::to_string(totalPixels));
    }
    
    // Assume the image data is stored as 16-bit unsigned ints.
    ImageStreamIO_semwait( &subarray, 1);  // waiting for image update

    int32_t* data = subarray.array.SI32;
    if (!data) {
        throw std::runtime_error("Data pointer in shared memory is null.");
    }
    Eigen::Map<const Eigen::Array<int32_t, Eigen::Dynamic, 1>> rawArr(data, totalPixels);
    Eigen::VectorXd frame = rawArr.cast<double>();

    // uint16_t* data = subarray.array.UI16;
    // if (!data) {
    //     //ImageStreamIO_destroyIm(&subarray);
    //     throw std::runtime_error("Data pointer in shared memory is null.");
    // }
    
    // // Convert the pixel data to an Eigen vector.
    // Eigen::VectorXd frame(totalPixels);
    // for (int i = 0; i < totalPixels; ++i) {
    //     frame(i) = static_cast<double>(data[i]);
    // }
    
    // Clean up the mapping.
    // ImageStreamIO_destroyIm(&subarray);
    
    return frame;
}




//BCB
// void updateDMSharedMemory(const Eigen::VectorXd &dmCmd) {
//     auto *md = dm_rtc.md;
//     const int N = md->nelement;
    

//     if ((int)dmCmd.size() != N) {
//         std::cerr << "DM command size mismatch: expected " 
//                   << N << ", got " << dmCmd.size() << "\n";
//         return;
//     }

//     // mark that we're writing
//     md->write = 1;

//     // copy the data
//     std::memcpy(dm_rtc.array.D, dmCmd.data(), N * sizeof(double));

//     // bump the update counter
//     md->cnt0++;
//     md->cnt1++; // could this be my type = 0;

//     // make sure all stores are visible before we wake the reader
//     std::atomic_thread_fence(std::memory_order_release);

//     //ImageStreamIO_sempost(&dm_rtc, -1)
//     ImageStreamIO_sempost(&dm_rtc0, 1); // post to master!! 
    
//     // 5) wake *only* semaphore #1 (the DM thread is waiting on index=1)
//     //if (ImageStreamIO_sempost(&dm_rtc, /*semid=*/1) != 0) {
//     //    std::cerr << "Error posting DM semaphore #1\n";
//     //}

//     // 6) clear the write flag
//     md->write = 0;
//}


// /**                     
// writes to DM SHM.
// **/
// void updateDMSharedMemory(const Eigen::VectorXd &dmCmd) {

//     // Get image parameters (assumed constant during runtime)
//     //int dm_naxis = dm_rtc.md->naxis;
//     //int dm_width  = dm_rtc.md->size[0];
//     //int dm_height = (naxis > 1) ? dm_rtc.md->size[1] : 1;
//     int dm_totalElements = dm_rtc.md->nelement; // e.g., 1600 for 40x40
    

//     // Check that dmCmd has the proper size.
//     if (dmCmd.size() != dm_totalElements) {
//         std::cerr << "DM command vector size mismatch: expected " 
//                   << dm_totalElements << ", got " << dmCmd.size() << std::endl;
//         return;
//     }
    
//     // Signal that we are writing.
//     dm_rtc.md->write = 1;
    
//     // Ensure the write‐flag store happens before the memcpy
//     std::atomic_thread_fence(std::memory_order_release);

//     // Use memcpy to copy the new DM command values into the shared memory.
//     // Ensure that the Eigen vector data is contiguous by using .data()
//     std::memcpy(dm_rtc.array.D, dmCmd.data(), dm_totalElements * sizeof(double));
    
//     // Update counters to notify consumers that data has been updated.
//     dm_rtc.md->cnt0++;  
//     dm_rtc.md->cnt1 = 0;
    
//     // Post the semaphore once to signal that new data is available.
//     int ret = ImageStreamIO_sempost(&dm_rtc, -1);
//     if (ret != 0) {
//         std::cerr << "Error posting semaphore: " << ret << std::endl;
//     }
    
//     // Clear the write flag.
//     dm_rtc.md->write = 0;
//}

// median helper to make bad_pixels = frame median (used in normalization) post TTonsky , 
namespace {
inline double median_of(const Eigen::VectorXd& v) {
    const auto N = static_cast<size_t>(v.size());
    if (N == 0) return 0.0;
    std::vector<double> tmp(v.data(), v.data() + v.size());
    const size_t mid = N / 2;
    std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
    double med_hi = tmp[mid];
    if ((N & 1) == 1) return med_hi;  // odd

    // even: need the max of the lower half
    auto lo_end = tmp.begin() + mid;
    auto lo_max_it = std::max_element(tmp.begin(), lo_end);
    return 0.5 * (med_hi + *lo_max_it);
}
} // namespace



// A helper function to print the dimension information of a matrix.
template <typename Derived>
void printMatrixDimensions(const std::string &name,
                             const Eigen::MatrixBase<Derived>& M) {
    std::cout << name << " dimensions: " 
              << M.rows() << " x " << M.cols() << std::endl;
}

// A helper function to print the size of a vector.
void printVectorSize(const std::string &name, const Eigen::VectorXd& v) {
    std::cout << name << " size: " << v.size() << std::endl;
}


// -- specific helpers for signal injection in command space 
namespace {

using clock_t = std::chrono::steady_clock;

static inline std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

struct InjCache {
    bool ready = false;

    // identity of what we cached for
    std::string basis_key;   // lower-cased basis name
    int         basis_index = -1;
    int         nAct = 0;

    // precomputed vectors
    Eigen::VectorXd p_all;   // nAct
    Eigen::VectorXd p_lo;    // nAct
    Eigen::VectorXd p_ho;    // nAct
    Eigen::VectorXd v_lo;    // size 2
    Eigen::VectorXd v_ho;    // size nHO
};

static InjCache g_inj_cache;

struct SignalInjRuntime {
    bool inited = false;

    // Basis vector (unit-normalized in command space, length = nAct)
    Eigen::VectorXd basis_vec;

    // PRBS state
    uint32_t lfsr = 0xACE1u;

    // White-noise RNG (unused here but handy if you add "white")
    std::mt19937_64 rng{0xdeadbeef};

    // Timing + frame cadence
    clock_t::time_point t0;
    long long frame_idx = 0;

    // Sample-and-hold + latency
    double last_sample = 0.0;     // held scalar
    std::deque<double> latency_q; // size = latency_frames
    // NEW: scalar actually sent this frame after latency
    double last_out_scalar = 0.0;      // <— added this when we moved to allow branch and space (not just dm) injection options
} g_inj;


/**
 * Build a command-space basis vector:

 */
inline void get_basis_mode(const bdr_rtc_config &rtc_config,
                           const std::string   &basis_name,
                           int                  basis_index,
                           Eigen::VectorXd     &basis_vec_out)
{
    const int nAct = static_cast<int>(rtc_config.zeroCmd.size());
    if (basis_vec_out.size() != nAct) basis_vec_out.resize(nAct);

    // lower-case key to match dm::get_basis convention
    std::string key = basis_name;
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    // Ensure basis file is loaded
    if (!dm::ensure_loaded_default()) {
        throw std::runtime_error("RTC get_basis_mode: failed to load default DM basis file.");
    }

    // Fetch basis list
    auto modes_ptr = dm::get_basis(key);
    if (!modes_ptr) {
        throw std::runtime_error("RTC get_basis_mode: basis \"" + basis_name + "\" not found.");
    }

    // Index validation
    const int nModes = static_cast<int>(modes_ptr->size());
    if (basis_index < 0 || basis_index >= nModes) {
        throw std::runtime_error(
            "RTC get_basis_mode: index " + std::to_string(basis_index) +
            " out of range for basis \"" + basis_name + "\" (0.." + std::to_string(nModes-1) + ")."
        );
    }

    // Size validation
    const Eigen::VectorXd &mode = (*modes_ptr)[static_cast<size_t>(basis_index)];
    if (mode.size() != nAct) {
        throw std::runtime_error(
            "RTC get_basis_mode: size mismatch for basis \"" + basis_name + "\" index " +
            std::to_string(basis_index) + ": mode has " + std::to_string(mode.size()) +
            " elements, expected " + std::to_string(nAct) + " (DM actuators)."
        );
    }

    // Use the raw mode exactly as stored (NO normalization)
    basis_vec_out = mode;
}


// Minimal Tikhonov-regularized LS solve: argmin_v ||A v - b||^2 + \lambda||v||^2
static Eigen::VectorXd tikhonov_ls(const Eigen::MatrixXd& A,
                                   const Eigen::VectorXd& b,
                                   double lambda = 1e-8)
{
    // (A.T A + lambda I) v = A.T b
    const int m = static_cast<int>(A.cols());
    Eigen::MatrixXd AtA = A.transpose() * A;
    AtA.diagonal().array() += lambda;
    Eigen::VectorXd Atb = A.transpose() * b;
    return AtA.ldlt().solve(Atb);
}

static void rebuild_injection_cache(const bdr_rtc_config& cfg)
{
    // Ensure basis_vec is current (no normalization!)
    get_basis_mode(cfg, cfg.inj_signal.basis, cfg.inj_signal.basis_index, g_inj.basis_vec);

    const int nAct = static_cast<int>(cfg.zeroCmd.size());
    g_inj_cache.ready       = false;         // will set true at end
    g_inj_cache.nAct        = nAct;
    g_inj_cache.basis_key   = to_lower_copy(cfg.inj_signal.basis);
    g_inj_cache.basis_index = cfg.inj_signal.basis_index;

    // Actuator-space target for injection
    g_inj_cache.p_all = g_inj.basis_vec;     // same size as DM

    // LO: project onto colspace(M2C_LO) and compute setpoint vector
    if (cfg.matrices.M2C_LO.rows() != nAct) {
        throw std::runtime_error("M2C_LO row mismatch vs zeroCmd size.");
    }
    // could add : if cfg.inj_signal.ls_method == "tikhonov" ...else other mthds
    g_inj_cache.v_lo = tikhonov_ls(cfg.matrices.M2C_LO, g_inj_cache.p_all, 1e-8); // size 2
    g_inj_cache.p_lo = cfg.matrices.M2C_LO * g_inj_cache.v_lo;                     // actuator equiv

    // HO: project onto colspace(M2C_HO) and compute setpoint vector
    if (cfg.matrices.M2C_HO.rows() != nAct) {
        throw std::runtime_error("M2C_HO row mismatch vs zeroCmd size.");
    }
    g_inj_cache.v_ho = tikhonov_ls(cfg.matrices.M2C_HO, g_inj_cache.p_all, 1e-8); // size nHO
    g_inj_cache.p_ho = cfg.matrices.M2C_HO * g_inj_cache.v_ho;                     // actuator equiv

    g_inj_cache.ready = true;
}


// Convert degrees to radians
inline double deg2rad(double d) { return d * M_PI / 180.0; }

// PRBS31: taps 31,28 -> returns ±1
inline int prbs31_step(uint32_t &l) {
    uint32_t bit = ((l >> 30) ^ (l >> 27)) & 1u;
    l = (l << 1) | bit;
    return (l & 1u) ? +1 : -1;
}

// Elapsed seconds since t0
inline double now_s() {
    static const auto t0 = clock_t::now();
    std::chrono::duration<double> dt = clock_t::now() - t0;
    return dt.count();
}


/// MOVED OUT OF namespace g_inj 
// // reset injection signal runtime
// void reset_signal_injection_runtime() {
//     g_inj = SignalInjRuntime{};  // zero-initialize: inited=false, clears FIFO/PRBS/frame_idx/basis_vec
//     //key point is inited becomes false -> so compute_c_inj(...) does its one-time init next frame
// }



/**
 * Sample the scalar carrier at relative time t (seconds) per the waveform.
 * amplitude multiplies the unit-normalized basis mode.
 */
inline double sample_waveform(const bdr_signal_cfg &cfg, double t)
{
    const double A   = cfg.amplitude;
    const double ph  = deg2rad(cfg.phase_deg);
    const double f   = cfg.freq_hz > 0.0 ? cfg.freq_hz : 0.0;
    const double dut = std::clamp(cfg.duty, 0.0, 1.0);

    if (cfg.waveform == "sine") {
        return A * std::sin(2.0 * M_PI * f * t + ph);
    }
    else if (cfg.waveform == "square") {
        if (f <= 0.0) return 0.0;
        const double T = 1.0 / f;
        const double tau = std::fmod(t + (ph / (2.0 * M_PI * f)), T); // phase shift as time
        return (tau / T < dut) ? A : 0.0;  // 0/A PWM style (not ±A)
    }
    else if (cfg.waveform == "step") {
        return A; // constant after gate opens
    }
    else if (cfg.waveform == "chirp") {
        // Linear chirp: f(t) = f0 + k t, k = (f1 - f0)/T
        // Phase φ(t) = 2π [ f0 t + 0.5 k t^2 ] + ph, limited to window [0, T]
        double T = std::max(cfg.chirp_T, 1e-6);
        double t_clip = std::clamp(t, 0.0, T);
        double k = (cfg.chirp_f1 - cfg.chirp_f0) / T;
        double phi = 2.0 * M_PI * (cfg.chirp_f0 * t_clip + 0.5 * k * t_clip * t_clip) + ph;
        return A * std::sin(phi);
    }
    else if (cfg.waveform == "prbs") {
        int s = prbs31_step(g_inj.lfsr); // ±1
        return A * static_cast<double>(s);
    }
    // "none" or unknown
    return 0.0;
}

/**
 * Compute c_inj (length nAct) for this frame:
 * - Gated by t_start_s / t_stop_s
 * - Sample-and-hold every hold_frames
 * - Latency in frames via a scalar FIFO
 */
inline void compute_c_inj(const bdr_rtc_config &rtc_config,
                          Eigen::Ref<Eigen::VectorXd> c_inj_out)
{
    const auto &cfg = rtc_config.inj_signal;           // <-- bdr_signal_cfg lives here
    const int nAct  = static_cast<int>(rtc_config.zeroCmd.size());

    // Default: no injection
    c_inj_out.setZero();

    if (!cfg.enabled) return;
    //if (cfg.space != "dm") return;             // so far we only implmented this for dm command space 

    // Lazy one-time init
    if (!g_inj.inited) {
        g_inj.t0 = clock_t::now();
        g_inj.frame_idx = 0;
        g_inj.lfsr = cfg.prbs_seed ? cfg.prbs_seed : 0xACE1u;
        g_inj.latency_q.clear();
        g_inj.latency_q.resize(static_cast<size_t>(std::max(0, cfg.latency_frames)), 0.0);
        get_basis_mode(rtc_config, cfg.basis, cfg.basis_index, g_inj.basis_vec);
        g_inj.inited = true;
    }

    // Time gating
    const double t_abs = now_s();
    if (t_abs < cfg.t_start_s) { 
        // still before start; shift latency queue with zeros to keep alignment
        if (!g_inj.latency_q.empty()) { g_inj.latency_q.pop_front(); g_inj.latency_q.push_back(0.0); }
        g_inj.frame_idx++;
        return;
    }
    if (cfg.t_stop_s > 0.0 && t_abs > cfg.t_stop_s) {
        // after stop; flush zeros through latency
        if (!g_inj.latency_q.empty()) { g_inj.latency_q.pop_front(); g_inj.latency_q.push_back(0.0); }
        g_inj.frame_idx++;
        return;
    }

    // Sample-and-hold: only refresh the scalar every 'hold_frames'
    double scalar = g_inj.last_sample;
    const int H = std::max(1, cfg.hold_frames);
    if ((g_inj.frame_idx % H) == 0) {
        const double t_rel = t_abs - cfg.t_start_s;
        scalar = sample_waveform(cfg, t_rel);
        g_inj.last_sample = scalar;
    }

    // Apply latency (scalar FIFO)
    double delayed_scalar = scalar;
    if (!g_inj.latency_q.empty()) {
        g_inj.latency_q.pop_front();
        g_inj.latency_q.push_back(scalar);
        delayed_scalar = g_inj.latency_q.front();
    }

    g_inj.last_out_scalar = delayed_scalar;   // <— added this when we moved to allow branch and space (not just dm) injection options

    // c_inj = delayed_scalar * basis_vec
    if (g_inj.basis_vec.size() != nAct) {
        // defensive (e.g. DM size changed)
        get_basis_mode(rtc_config, cfg.basis, cfg.basis_index, g_inj.basis_vec);
    }
    c_inj_out = delayed_scalar * g_inj.basis_vec;


    g_inj.frame_idx++;
}

} // namespace


void reset_signal_injection_runtime() {
    g_inj = SignalInjRuntime{};  // zero-initialize: inited=false, clears FIFO/PRBS/frame_idx/basis_vec
    //key point is inited becomes false -> so compute_c_inj(...) does its one-time init next frame
}

// // The main RTC function
void rtc(){
    // Temporary variabiles that don't need initialisation
    // can go here.


    // std::cout << "Controller kp size: " << rtc_config.controller.kp.size() << std::endl;
    // std::cout << "Controller ki size: " << rtc_config.controller.ki.size() << std::endl;
    // std::cout << "Controller kd size: " << rtc_config.controller.kd.size() << std::endl;
    // std::cout << "Controller lower_limits size: " << rtc_config.controller.lower_limits.size() << std::endl;
    // std::cout << "Controller upper_limits size: " << rtc_config.controller.upper_limits.size() << std::endl;
    // std::cout << "Controller set_point size: " << rtc_config.controller.set_point.size() << std::endl;

    std::cout << "ctrl I2M_LO size: " << rtc_config.matrices.I2M_LO.size() << std::endl;
    std::cout << "ctrl I2M_HO size: " << rtc_config.matrices.I2M_HO.size() << std::endl;

    std::cout << "ctrl M2C_HO size: " << rtc_config.matrices.M2C_LO.size() << std::endl;
    std::cout << "ctrl M2C_HO size: " << rtc_config.matrices.M2C_HO.size() << std::endl;

    // comment out while upgrading controller class 20-8-25. Uncomment later after verified (may need to change some things)
    // std::cout << "ctrl kp LO size: " << rtc_config.ctrl_LO.kp.size() << std::endl;
    // std::cout << "ctrl kp size: " << rtc_config.ctrl_HO.kp.size() << std::endl;
    // std::cout << "Controller ki size: " << rtc_config.ctrl_LO.ki.size() << std::endl;
    // std::cout << "Controller kd size: " << rtc_config.ctrl_LO.kd.size() << std::endl;
    // std::cout << "Controller lower_limits size: " << rtc_config.ctrl_LO.lower_limits.size() << std::endl;
    // std::cout << "Controller upper_limits size: " << rtc_config.ctrl_LO.upper_limits.size() << std::endl;
    // std::cout << "Controller set_point size: " << rtc_config.ctrl_LO.set_point.size() << std::endl;

    std::cout << "M2C_HO" << rtc_config.matrices.M2C_HO.size() << std::endl;

    std::cout << "secondary pixels size: " << rtc_config.pixels.secondary_pixels.size() << std::endl;
    std::cout << "exterior pixels size: " << rtc_config.pixels.exterior_pixels.size() << std::endl;

    std::cout << "secondary strehl matrix: " << rtc_config.matrices.I2rms_sec << std::endl;
    std::cout << "exterior strehl matrix: " << rtc_config.matrices.I2rms_ext << std::endl;
    
    std::cout << "szm: " << rtc_config.matrices.szm << std::endl;
    std::cout << "sza: " << rtc_config.matrices.sza << std::endl;
    std::cout << "szp: " << rtc_config.matrices.szp << std::endl;

    std::cout << "I2A.cols() : " << rtc_config.matrices.I2A.cols() << std::endl;
    std::cout << "I2A.rows() : " << rtc_config.matrices.I2A.rows() << std::endl;


    if (dm_rtc.md) {
        int dm_naxis = dm_rtc.md->naxis;
        int dm_width = dm_rtc.md->size[0];
        int dm_height = (dm_naxis > 1) ? dm_rtc.md->size[1] : 1;
        int dm_totalElements = dm_rtc.md->nelement; //  totalElements = width * height
        std::cout << "Shared memory size: " << dm_width << " x " << dm_height << " pixels ("
                << dm_totalElements << " elements)" << std::endl;
    } else {
        std::cerr << "Error: No metadata available in the dm shared memory image." << std::endl;
    }



    if (subarray.md) {
        int img_naxis = subarray.md->naxis;
        int img_width = subarray.md->size[0];
        int img_height = (img_naxis > 1) ? subarray.md->size[1] : 1;
        int img_totalElements = subarray.md->nelement; //  totalElements = width * height
        std::cout << "Shared memory size: " << img_width << " x " << img_height << " pixels ("
                << img_totalElements << " elements)" << std::endl;
    } else {
        //std::cout << "here" << std::endl;
        std::cerr << "Error: No metadata available in the subarray shared memory image." << std::endl;
    }


    
    //// getting telemetry in AIV 

    // std::cerr << "setting up telemetery SHM for offline plotting" << std::endl;

    // // Allocate shapes
    // uint32_t size_sig[2] = {140, shm_telem_samples};
    // uint32_t size_eLO[2] = {2, shm_telem_samples};
    // uint32_t size_eHO[2] = {140, shm_telem_samples};
    // // Create SHM images with no semaphores and no keywords
    // ImageStreamIO_createIm(&shm_sig, "sig_telem", 2, size_sig, _DATATYPE_FLOAT, 1, 0);
    // ImageStreamIO_createIm(&shm_eLO, "eLO_telem", 2, size_eLO, _DATATYPE_FLOAT, 1, 0);
    // ImageStreamIO_createIm(&shm_eHO, "eHO_telem", 2, size_eHO, _DATATYPE_FLOAT, 1, 0);




    ////////////////////////////////////////////////////
    /// parameters below are calculated at run time (derived from rtc_config) so 
    // I made a method in the  bdr_rtc_config struct to init these..
    ////////
    //// Aspects to be re-computed everytime the configuration changes - init_rtc function? 
    // if (rtc_config.state.controller_type=="PID"){
    //     ////heree
    //     // PIDController ctrl_LO( rtc_config.controller );
    //     // PIDController ctrl_HO( rtc_config.controller ); // this will have to use different configs / lengths !  
    //     ctrl_LO = PIDController( rtc_config.ctrl_LO_config );
    //     ctrl_HO = PIDController( rtc_config.ctrl_HO_config ); // this will have to use different configs / lengths !  
    //     }
    // else{
    //     std::cout << "no ctrl match" << std::endl;
    // }


    // // some pre inits
    // Eigen::VectorXd img = Eigen::VectorXd::Zero(rtc_config.matrices.szp); // P must be defined appropriately.
    // Eigen::VectorXd zeroCmd = Eigen::VectorXd::Zero(rtc_config.matrices.sza);

    // ----------------------- IMPORTANT 
    // frames per second from rtc_config used for converting dark from adu/s -> adu 
    // we should actually read this from the current camera settings when calling the rtc! 
    // double fps = std::stof(rtc_config.cam.fps);
    // double gain = std::stof(rtc_config.cam.gain);
    // double scale = gain / fps;
    // // ALL I2M and reference intensities stored in config are in ADU/second/gain !!
    // Eigen::MatrixXd I2M_LO = scale * rtc_config.matrices.I2M_LO;
    // Eigen::MatrixXd I2M_HO = scale * rtc_config.matrices.I2M_HO;
    
    // Eigen::VectorXd N0_dm =  scale * rtc_config.reference_pupils.norm_pupil_dm;
    // Eigen::VectorXd I0_dm =  scale * rtc_config.reference_pupils.I0_dm;

    // // darks (generated in dcs/calibration_frames/gen_dark_bias_badpix.py ) are adu/s in the gain setting (not normalized by gain)
    // // I should review the darks and perhaps normalize by gain setting too! 
    // Eigen::VectorXd dark_dm = 1.0 / fps * rtc_config.reduction.dark_dm;



    // Strehl models
    // secondary pixel. Solarstein mask closer to UT size - this will be invalid on internal source 
    // int sec_idx = rtc_config.pixels.secondary_pixels(4); // .secondary_pixels defines 3x3 square around secondary - we only use the central one
    // double m_s = scale * rtc_config.matrices.I2rms_sec(0, 0); // intensity is normalized by fps and gain in model (/home/asg/Progs/repos/asgard-alignment/calibration/build_strehl_model.py)
    // double b_s = rtc_config.matrices.I2rms_sec(1, 1);
    
    //exterior pixels 
    //double m_e = rtc_config.matrices.I2rms_sec(0, 0);
    //double b_e = rtc_config.matrices.I2rms_sec(1, 1);


    // try this 
    // auto A = rtc_config.I2M_LO_runtime;  // size 2×P
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU);
    // Eigen::Matrix2d U = svd.matrixU();


    // naughty actuator or modes 
    std::vector<int> naughty_list(140, 0);

    // time stuff 
    double current_time_ms;
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // init disturbance injection command
    c_inj.resize(rtc_config.zeroCmd.size());
    c_inj.setZero();
    
    //static open loop offsets in LO and HO 
    ol_lo.resize(rtc_config.zeroCmd.size());
    ol_ho.resize(rtc_config.zeroCmd.size());
    ol_lo.setZero();
    ol_ho.setZero();
    ///////////////////////////////////////


    /// DERIVED PARAMETERS FROM rtc_config
    // {
    //     std::scoped_lock lock(ctrl_mutex, telemetry_mutex); //C++17
    //     rtc_config.initDerivedParameters();
    // }
    //////////////////////////////////////

    /// this can cause problems and should be automatic!!!! 
    const int expectedPixels = rtc_config.matrices.I2A.cols(); //31 * 31;

    // Pre‑compute totalPixels so we only do it once
    int nx = subarray.md->size[0];
    int ny = (subarray.md->naxis > 1) ? subarray.md->size[1] : 1;
    int totalPixels = nx * ny;
    if (totalPixels != expectedPixels) {
        std::cout << "totalPixels" << totalPixels << std::endl;
        std::cout << "expectePixels" << expectedPixels << std::endl; 
        throw std::runtime_error("RTC: shared‑memory geometry mismatch");
    }
    std::cout << "RTC CTRL TYPE" << rtc_config.state.controller_type << std::endl;



    // pick an arbitrary preferred sem index (0 is fine)
    int semid = 1 ; //ImageStreamIO_getsemwaitindex(&subarray, /*preferred*/ 0);
    //int semid_dm = ImageStreamIO_getsemwaitindex(&dm_rtc0, /*preferred*/ 1);

    // LOOP SPEED (us) - with camera sem this isn not necessary but can be used for testing ! 
    //std::chrono::microseconds loop_time( 1000 ); //static_cast<long long>(1.0/fps * 1e6) ); // microseconds - set speed of loop 
    //constexpr auto loop_time = std::chrono::microseconds( loop_time ); //1400); //(5000); //  200Hz #1 kHz
    //auto next_tick = std::chrono::steady_clock::now();

    // ------------------- to try
    // //to try 
    // // online calibration of TT modal leakage
    // auto A = rtc_config.I2M_LO_runtime;      // 2×P interaction matrix
    // Eigen::Matrix2d D = Eigen::Matrix2d::Identity();  // decoupling matrix

    // // tone injection parameters
    // constexpr double inj_freq = 10.0;      // Hz
    // constexpr double inj_amp  = 0.008;     // small amplitude
    // double phase  = 0.0;
    // double dt_s = loop_time.count() * 1e-6;            // loop_time in seconds

    // // demod accumulators
    // std::complex<double> Z_tt{0,0}, Z_tl{0,0};
    // double norm_c = 0.0;
    // int calibCount = 0;
    // ------------------- end to try

    // before entering main loop, drain any stale posts:
    catch_up_with_sem(&subarray, semid);
    catch_up_with_sem(&dm_rtc0, 1); // necessary?


    std::cout << "RTC started" << std::endl;
    while(servo_mode.load() != SERVO_STOP){

        //std::cout << servo_mode_LO << std::endl;
        //std::cout << servo_mode_HO << std::endl;
        // Check if we need to pause the loop.
        if (pause_rtc.load()) {
            // Use a unique_lock with the mutex for the condition variable.
            // (A condition variable’s wait functions require a std::unique_lock<std::mutex> rather than a std::lock_guard<std::mutex> because the condition variable needs the flexibility to unlock and later re-lock the mutex during the wait. )
            std::unique_lock<std::mutex> lock(rtc_pause_mutex);
            // Wait until pause_rtc becomes false.
            // This lambda serves as a predicate.
            rtc_pause_cv.wait(lock, [](){ return !pause_rtc.load(); });
            // catch up on any missed semaphore posts exactly once after pause
            catch_up_with_sem(&subarray, semid);
            std::cout<< "caught up with semaphores amd resuming" << std::endl;
            //just_resumed = true;
        }

        // catch up on any missed semaphore posts exactly once after pause
        // if (just_resumed) {
        //     //ImageStreamIO_semflush(&subarray, -1); // maybe this but not clear
        //     catch_up_with_sem(&subarray, semid);
        //     just_resumed = false;
        //     std::cout<< "flushed semaphores amd resuming" << std::endl;
        // }

        // if we want to measure latency including wainting for neew frame 
        //start = std::chrono::steady_clock::now();


        //img =  getFrameFromSharedMemory(32*32); //32*32);
        
        //ImageStreamIO_semwait(&subarray, /*timeout=*/-1);

        ImageStreamIO_semwait(&subarray, 1); //semid);

        // to measure calculation time (seems around 110us)
        //start = std::chrono::steady_clock::now();
        
        // check skipped frames 
        //std::cout << subarray.md->cnt0 << std::endl;

        //  MAP & COPY into an Eigen vector
        //    (we know data is stored U16 so we cast to uint16_t*)
        // uint16_t *raw = subarray.array.UI16;
        // // Eigen::VectorXd img(totalPixels);
        // // for (int i = 0; i < totalPixels; ++i) {
        // //     img(i) = static_cast<double>(raw[i]);
        // // }
        // // a better way 
        // // Map the raw uint16_t buffer as an Eigen array:
        // Eigen::Map<const Eigen::Array<uint16_t,Eigen::Dynamic,1>> rawArr(raw, totalPixels);

        // // Cast it to double—and store into a VectorXd:
        // Eigen::VectorXd img = rawArr.cast<double>();
        
        //-----1. S T A R T  C L O C K -----//
        // auto start1 = Clock::now();

        int32_t *raw = subarray.array.SI32;
        Eigen::Map<const Eigen::Array<int32_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
        Eigen::VectorXd img = rawArr.cast<double>(); // img IS ALWAYS RAW , AND WHAT WE KEEP IN TELEMETRY. WE NEED THIS TO MAKE DARKS ETC

        // images have 1000 aduoffset. We could subtract this here.. but better to have a live background estimate
        //img -= 1000.0; // quick and dirty background subtraction - should be dark + bias
        // 18/9/2025
        if (rtc_config.reduction.dark.size() != img.size()) {
            rtc_config.reduction.dark = Eigen::VectorXd::Constant(img.size(), 1000.0);
        }
        Eigen::VectorXd img_red = img - rtc_config.reduction.dark;


        // --- Bad-pixel repair -> subframe flux -> normalization ---
        // If we have bad pixels listed, set those indices to the frame median BEFORE summing.
        // post TTonsky
        if (rtc_config.pixels.bad_pixels.size() > 0) {
            const double med = median_of(img_red);
            const auto& bp = rtc_config.pixels.bad_pixels; // Eigen::Matrix<int32_t, Dyn, 1>
            for (Eigen::Index k = 0; k < bp.size(); ++k) {
                const int32_t idx = bp(k);
                if (idx >= 0 && idx < img.size()) {
                    img_red(idx) = med;
                }
            }
        }

        // Compute subframe flux (sum of intensities) and store it globally.
        //g_subframe_int = img.sum();
        g_subframe_int = img_red.sum();   // was img.sum()

        // ===== New: skip this RTC iteration if flux is invalid =====
        bool bad_flux_frame = (!std::isfinite(g_subframe_int) || g_subframe_int <= 0.0);
        if (bad_flux_frame) {
            static uint64_t bad_flux_skip_count = 0;
            ++bad_flux_skip_count;

            // Log first few and then every 100th to avoid spam
            if (bad_flux_skip_count <= 5 || (bad_flux_skip_count % 100 == 0)) {
                std::cerr << "[RTC] Skipping frame due to invalid subframe sum: "
                        << g_subframe_int << "  (skip_count=" << bad_flux_skip_count << ")\n";
            }

            // Nothing else this iteration: do NOT normalize, do NOT project, do NOT update telem/ctrl
            // (If you need to record a 'dropped' mark in telemetry, do it here.)
            //continue;  // <-- EARLY EXIT from this RTC loop iteration
        }

        // // Guard against pathological cases (negative/zero/NaN/Inf)
        // if (!std::isfinite(g_subframe_int) || g_subframe_int <= 0.0) {
        //     g_subframe_int = 1.0;
        //     // need to set a flag so the rtc skips this frame
        // }

        // we dont Normalize the image by subframe intensity sum
	// we do this in the img_dm from which we calculate the signal
	// this way we can extract flux sum from the image for later analysis 

	// img /= g_subframe_int;

	// --- End normalization patch ---


        // auto end1   = Clock::now();
        // auto us1    = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count();
        // std::cout << "[TIMER] Process image took " << us1 << " ns\n";
        //----- E N D  C L O C K -----//

        //std::cout  << img.size() << std::endl;
        //uint64_t totalFramesWritten = subarray.md->cnt0;
        //uint64_t lastSliceIndex     = ImageStreamIO_readLastWroteIndex(&subarray);

        // // Print them
        // std::cout 
        //     << "Frame #" << totalFramesWritten 
        //     << " landed in buffer slot " << lastSliceIndex 
        //     << std::endl;
        
        //-----2. C L O C K -----//
        // auto start2 = Clock::now();
        
        // model of residual rms in DM units using secondary obstruction 
        //dm_rms_est_1 = img[ rtc_config.pixels.exterior_pixels ].template cast<double>().mean(); //rtc_config.m_s_runtime * ( img[  rtc_config.sec_idx ] - rtc_config.reduction.dark[ rtc_config.sec_idx ] - rtc_config.reduction.bias[ rtc_config.sec_idx ] ) +  rtc_config.b_s_runtime;
        // new using average of all exterior pixels.
        // double sum_ext_px = 0.0;
        // for (Eigen::Index k = 0; k < rtc_config.pixels.exterior_pixels.size(); ++k) {
        //     sum_ext_px += img(rtc_config.pixels.exterior_pixels(k));
        // }
        // double dm_rms_est_1 = (rtc_config.pixels.exterior_pixels.size() > 0)
        //     ? sum_ext_px / static_cast<double>(rtc_config.pixels.exterior_pixels.size())
        //     : std::numeric_limits<double>::quiet_NaN();

        // 18/9/2025 - do it on reduced image (dark subtracted)
        double sum_ext_px = 0.0;
        for (Eigen::Index k = 0; k < rtc_config.pixels.exterior_pixels.size(); ++k) {
            sum_ext_px += img_red(rtc_config.pixels.exterior_pixels(k));  // was img(...)
        }
        double dm_rms_est_1 = (rtc_config.pixels.exterior_pixels.size() > 0)
            ? sum_ext_px / static_cast<double>(rtc_config.pixels.exterior_pixels.size())
            : std::numeric_limits<double>::quiet_NaN();

        //std::cout << dm_rms_est_1 << std::endl;

        // go to dm space subtracting dark (ADU/s) and bias (ADU) there
        // should actually read the current fps rather then get it from config file
	    // HERE WE DO OUR NORMALIZATION 
        //img_dm = (rtc_config.matrices.I2A *  img) / g_subframe_int ; //-  rtc_config.dark_dm_runtime - rtc_config.reduction.bias_dm; //1 / fps * rtc_config.reduction.dark_dm;

        img_dm = (rtc_config.matrices.I2A * img_red) / g_subframe_int; //18/9/2025

        //sig = (img_dm - rtc_config.I0_dm_runtime).cwiseQuotient(rtc_config.N0_dm_runtime); //(img_dm - rtc_config.reference_pupils.I0_dm).cwiseQuotient(rtc_config.reference_pupils.norm_pupil_dm);
        

        // auto end2   = Clock::now();
        // auto us2    = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
        // std::cout << "[TIMER] I2A projection took " << us2 << " ns\n";
        //----- E N D  C L O C K -----//
        
        

        //-----3. C L O C K -----//
        // auto start3 = Clock::now();
        
        //  Compute the averaged signal, telem size can change mid rtc so check! 
        size_t M = rtc_config.telem.signal.size();

        // added correct scaling to scale the reference relative to the current subframe flux
        double flux_scaling = g_subframe_int / rtc_config.reference_pupils.intrn_flx_I0 ; // we want to keep the signal in flux units (adu/s) so we multiply by subframe sum here
	// remove flux scaling mult.
        sig = (img_dm - rtc_config.I0_dm_runtime).cwiseQuotient(rtc_config.N0_dm_runtime); //(img_dm - rtc_config.reference_pupils.I0_dm).cwiseQuotient(rtc_config.reference_pupils.norm_pupil_dm);
        
        // add to telemetry! 
        rtc_config.telem.signal.push_back(sig);       // 'sig' is Eigen::VectorXd
        
        // uncomment and build July 2025 AIV
        //this boxcar weighted average is a first attempt - later evolve to
        //burst window in full unfiltered non-destructive read mode with slope est. 
        int boxcar = global_boxcar.load();
        if ((boxcar > 1) && (M > boxcar)) {
            sig = weightedAverage(rtc_config.telem.signal, boxcar);
        }

        
        // auto end3   = Clock::now();
        // auto us3    = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count();
        // std::cout << "[TIMER] signal and boxcar took " << us3 << " ns\n";
        //----- E N D  C L O C K -----//
        
        

        // ------------- START OF SIGNAL INJECTION SECTION -----------
        // ============================================================

        /* in rtc_config we have a nested bdr_signal_cfg struc which defined 
        what and where signals are injected into the controller. Either this can
        be in HO, LO or all branches and can be applied to the plant (command space)
        or to the setpoints. Here we check if cached signals need to be updated */

        //-----4. C L O C K -----//
        // auto start4 = Clock::now();

        // React to config changes
        static uint64_t seen_epoch = 0;
        if (g_inj_cfg_epoch.load(std::memory_order_relaxed) != seen_epoch) {
            seen_epoch = g_inj_cfg_epoch.load(std::memory_order_relaxed);
            reset_signal_injection_runtime();   // clears runtime (basis cached in g_inj)
            g_inj_cache.ready = false;          // force cache rebuild on use
        }

        // Compute this frame’s carrier once (don’t add to dm yet)
        Eigen::VectorXd ctmp(rtc_config.zeroCmd.size());
        compute_c_inj(rtc_config, ctmp);        // fills g_inj.last_out_scalar; ctmp not used directly

        // Rebuild cache if basis/basis_index changed
        const auto basis_key_now = to_lower_copy(rtc_config.inj_signal.basis);
        if (!g_inj_cache.ready ||
            g_inj_cache.basis_index != rtc_config.inj_signal.basis_index ||
            g_inj_cache.basis_key   != basis_key_now) {
            rebuild_injection_cache(rtc_config);   // computes p_all/p_lo/p_ho and v_lo/v_ho
        }

        // If injecting at SETPOINT, update controller setpoints BEFORE processing
        Eigen::VectorXd du_lo, du_ho;                 // for telemetry/safety later
        Eigen::VectorXd c_inj_applied = Eigen::VectorXd::Zero(ctmp.size()); // actuator-space equiv

        if (rtc_config.inj_signal.enabled &&
            rtc_config.inj_signal.apply_to == "setpoint")
        {
            const double a = g_inj.last_out_scalar;
            std::lock_guard<std::mutex> lock(ctrl_mutex);

            if ((rtc_config.inj_signal.branch == "LO" || rtc_config.inj_signal.branch == "ALL") &&
                servo_mode_LO.load() == SERVO_CLOSE)
            {
                du_lo = a * g_inj_cache.v_lo;
                const auto sp = std::get<Eigen::VectorXd>(rtc_config.ctrl_LO->get_parameter("set_point"));
                Eigen::VectorXd new_sp = (sp + du_lo).eval();                 // force materialization
                rtc_config.ctrl_LO->set_parameter("set_point", new_sp);

                // actuator-equivalent for telemetry only:
                c_inj_applied += rtc_config.matrices.M2C_LO * du_lo;
            }

            if ((rtc_config.inj_signal.branch == "HO" || rtc_config.inj_signal.branch == "ALL") &&
                servo_mode_HO.load() == SERVO_CLOSE)
            {
                du_ho = a * g_inj_cache.v_ho;
                const auto sp = std::get<Eigen::VectorXd>(rtc_config.ctrl_HO->get_parameter("set_point"));
                Eigen::VectorXd new_sp = (sp + du_ho).eval();                 // force materialization
                rtc_config.ctrl_HO->set_parameter("set_point", new_sp);

                c_inj_applied += rtc_config.matrices.M2C_HO * du_ho;
            }
        }

        // auto end4   = Clock::now();
        // auto us4    = std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - start4).count();
        // std::cout << "[TIMER] signal injection checks took " << us4 << " ns\n";
        //----- E N D  C L O C K -----//

        // if (rtc_config.inj_signal.enabled &&
        //     rtc_config.inj_signal.apply_to == "setpoint")
        // {
        //     const double a = g_inj.last_out_scalar;

        //     std::lock_guard<std::mutex> lock(ctrl_mutex);  // protect ctrl_* parameter updates

        //     if (rtc_config.inj_signal.branch == "LO" || rtc_config.inj_signal.branch == "ALL") {
        //         du_lo = a * g_inj_cache.v_lo;            // size 2
        //         // Update setpoint for LO BEFORE ctrl_LO->process(e_LO)
        //         auto sp = std::get<Eigen::VectorXd>(rtc_config.ctrl_LO->get_parameter("set_point"));
        //         rtc_config.ctrl_LO->set_parameter("set_point", sp + du_lo);

        //         // actuator-equivalent for telemetry/safety
        //         c_inj_applied += rtc_config.matrices.M2C_LO * du_lo;
        //     }

        //     if (rtc_config.inj_signal.branch == "HO" || rtc_config.inj_signal.branch == "ALL") {
        //         du_ho = a * g_inj_cache.v_ho;            // size nHO
        //         auto sp = std::get<Eigen::VectorXd>(rtc_config.ctrl_HO->get_parameter("set_point"));
        //         rtc_config.ctrl_HO->set_parameter("set_point", sp + du_ho);

        //         c_inj_applied += rtc_config.matrices.M2C_HO * du_ho;
        //     }
        // } 


        // ------------- END OF SIGNAL INJECTION SECTION -----------
        // ============================================================


        //-----5. C L O C K -----//
        // auto start5 = Clock::now();

        // open loop static offsets - distint from injection signals which are dynamic and applied in different places (e.g. setpoints)
        ol_lo.setZero(); // set zero each iteration and only populate if in open loop and offsets exist
        ol_ho.setZero();  
        auto off = std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
        if (off) {
            if (servo_mode_LO.load() == SERVO_OPEN) ol_lo = off->lo;
            if (servo_mode_HO.load() == SERVO_OPEN) ol_ho = off->ho;
        }

        // auto end5   = Clock::now();
        // auto us5    = std::chrono::duration_cast<std::chrono::nanoseconds>(end5 - start5).count();
        // std::cout << "[TIMER] checking static OL signal took " << us5 << " ns\n";
        //----- E N D  C L O C K -----//

        //-----6. C L O C K -----//
        auto start6 = Clock::now();

        //  Project into LO/HO as before, but now using the smoother sig_avg
        e_LO = rtc_config.I2M_LO_runtime * sig;

        e_HO = rtc_config.I2M_HO_runtime * sig;


        // auto end6   = Clock::now();
        // auto us6    = std::chrono::duration_cast<std::chrono::nanoseconds>(end6 - start6).count();
        // std::cout << "[TIMER] error calc took " << us6 << " ns\n";
        //----- E N D  C L O C K -----//

        //-------------------------------
        // try this 
        // //   Compute the raw tip/tilt errors
        // Eigen::Vector2d e_raw = rtc_config.I2M_LO_runtime * sig;

        // //   Inject a little sine tone onto the tip channel
        // double inj = inj_amp * std::sin(phase);
        // e_raw(0) += inj;
        // phase = std::fmod(phase + 2*M_PI*inj_freq*dt_s, 2*M_PI);

        // //  Demodulate both channels at inj_freq
        // double c = std::cos(phase);
        // Z_tt  += e_raw(0) * c;  // in‑phase tip response
        // Z_tl  += e_raw(1) * c;  // in‑phase tilt “leakage”
        // norm_c += c*c;

        // //   Every 1 k frames, update your C→D calibration
        // if (++calibCount == 1000) {
        //     // build static 2×2 gain matrix from demodulated tips
        //     Eigen::Matrix2d C;
        //     C << Z_tt.real()/norm_c,  Z_tl.real()/norm_c,
        //         Z_tl.real()/norm_c,  Z_tt.real()/norm_c;  
        //     D = C.inverse();              // new decoupling matrix

        //     // reset accumulators
        //     Z_tt.setZero(); Z_tl.setZero();
        //     norm_c = 0.0;  calibCount = 0;
        // }

        // //  2.5) Apply decoupling to your error vector
        // e_LO = D * e_raw;



        // 3) continue with controller.process(e_LO), etc.
        //e_LO =  rtc_config.I2M_LO_runtime * sig; //rtc_config.matrices.I2M_LO * sig ;

        //e_HO =  rtc_config.I2M_HO_runtime * sig; //rtc_config.matrices.I2M_HO * sig ;


        // if auto mode change state based on signals


        //-----7. C L O C K -----//
        // auto start7 = Clock::now();

        // LO CALCULATIONS 
        if (servo_mode_LO.load() == SERVO_CLOSE){
            if (!rtc_config.telem.LO_servo_mode.empty() && rtc_config.telem.LO_servo_mode.back() != SERVO_CLOSE) {
                std::cout << "LO IN CLOSED LOOP" << std::endl;

            }
            
            std::lock_guard<std::mutex> lock(ctrl_mutex);

            //std::cout << "HERE NOW, LO IN CLOSED LOOP" << std::endl;

            u_LO = rtc_config.ctrl_LO->process( e_LO );

            c_LO = rtc_config.matrices.M2C_LO * u_LO;

            //std::cout << "max e_LO" << e_LO.cwiseAbs().maxCoeff() << std::endl;
            //std::cout << "max u_LO" << u_LO.cwiseAbs().maxCoeff() << std::endl;
            //std::cout << "max C_LO" << c_LO.cwiseAbs().maxCoeff() << std::endl;

        
        }else if(servo_mode_LO.load() == SERVO_OPEN){
            if (!rtc_config.telem.LO_servo_mode.empty() && rtc_config.telem.LO_servo_mode.back() != SERVO_OPEN) {
                std::cout << "reseting LO controller" << std::endl;
                //reset controllers
                rtc_config.ctrl_LO->reset();
            }
  

            u_LO = 0 * e_LO;
            
            c_LO = rtc_config.matrices.M2C_LO * u_LO;

            


            //c_LO = rtc_config.matrices.M2C_LO * u_LO;
            
        }

        // HO CALCULATIONS 
        if (servo_mode_HO.load() == SERVO_CLOSE){
            if (!rtc_config.telem.HO_servo_mode.empty() && rtc_config.telem.HO_servo_mode.back() != SERVO_CLOSE) {
                std::cout << "HO IN CLOSED LOOP" << std::endl;

            }

            u_HO = rtc_config.ctrl_HO->process( e_HO );
            
            c_HO = rtc_config.matrices.M2C_HO * u_HO;
            
            //std::cout << c_HO.cwiseAbs().maxCoeff() << std::endl; 
            //std::cout << rtc_config.matrices.M2C_HO.cwiseAbs().maxCoeff() << std::endl; 
            //std::cout << sig.cwiseAbs().maxCoeff() << std::endl; 

        }else if(servo_mode_HO.load() == SERVO_OPEN){
            if (!rtc_config.telem.HO_servo_mode.empty() && rtc_config.telem.HO_servo_mode.back() != SERVO_OPEN) {
                std::cout << "reseting HO controller" << std::endl;
                //reset controllers
                rtc_config.ctrl_HO->reset();
            }
  

            u_HO = 0 * e_HO ;
            c_HO = rtc_config.matrices.M2C_HO * u_HO;

            
            
            //c_HO = rtc_config.matrices.M2C_HO * u_HO;
            
        }

        // auto end7   = Clock::now();
        // auto us7    = std::chrono::duration_cast<std::chrono::nanoseconds>(end7 - start7).count();
        // std::cout << "[TIMER] ctrl signal processing took " << us7 << " ns\n";

        //----- E N D  C L O C K -----//


        // === HO Safety Check: Zero gains for misbehaving actuators ===
        const double ho_threshold = 0.1; // around 700nm! assuming 7000nm/DM cmd unit

        if (u_HO.cwiseAbs().maxCoeff() > ho_threshold) {
            bool disabled_this_frame = false;

            for (int i = 0; i < u_HO.size(); ++i) {
                if (std::abs(u_HO(i)) > ho_threshold) {
                    //std::cout << "[SAFETY] HO actuator " << i << " exceeded threshold with value " << u_HO(i) << std::endl;
                    //// 20-8-25
                    // // Reset controller state
                    // rtc_config.ctrl_HO.integrals(i) = 0.0;
                    // rtc_config.ctrl_HO.prev_errors(i) = 0.0;
                    {
                    auto I = std::get<Eigen::VectorXd>(rtc_config.ctrl_HO->get_parameter("integrals"));
                    auto E = std::get<Eigen::VectorXd>(rtc_config.ctrl_HO->get_parameter("prev_errors"));
                    I(i) = 0.0;
                    E(i) = 0.0;
                    rtc_config.ctrl_HO->set_parameter("integrals", I);
                    rtc_config.ctrl_HO->set_parameter("prev_errors", E);
                    }

                    // Track misbehavior
                    naughty_list[i]++;

                    // Disable gains only once when crossing threshold
                    if (naughty_list[i] == 100) {
                        std::cout << "[SAFETY] Disabling gains for actuator " << i << " after 100 violations" << std::endl;

                        //// 20-8-25
                        // rtc_config.ctrl_HO.kp(i) = 0.0;
                        // rtc_config.ctrl_HO.ki(i) = 0.0;
                        // rtc_config.ctrl_HO.kd(i) = 0.0;

                        {
                        auto kp = std::get<Eigen::VectorXd>(rtc_config.ctrl_HO->get_parameter("kp"));
                        auto ki = std::get<Eigen::VectorXd>(rtc_config.ctrl_HO->get_parameter("ki"));
                        auto kd = std::get<Eigen::VectorXd>(rtc_config.ctrl_HO->get_parameter("kd"));
                        kp(i) = ki(i) = kd(i) = 0.0;
                        rtc_config.ctrl_HO->set_parameter("kp", kp);
                        rtc_config.ctrl_HO->set_parameter("ki", ki);
                        rtc_config.ctrl_HO->set_parameter("kd", kd);
                        }

                        disabled_this_frame = true;
                    }
                }
            }

            // If any actuator was just disabled (not already at 0), zero DM once
            if (disabled_this_frame) {
                std::cout << "[SAFETY] Zeroing DM due to newly disabled actuator" << std::endl;
                std::cout << "[SAFETY] Re-locking the loop" << std::endl;
                updateDMSharedMemory(dm_rtc, rtc_config.zeroCmd);
                ImageStreamIO_sempost(&dm_rtc0, 1);
            }
        }

        
        if (c_LO.cwiseAbs().maxCoeff() > 0.3) {
            // 20-8-25
            // Reset controller state
            // for (int i = 0; i < u_LO.size(); ++i) {
            //     rtc_config.ctrl_LO.integrals(i) = 0.0;
            //     rtc_config.ctrl_LO.prev_errors(i) = 0.0;
            // }
            std::lock_guard<std::mutex> lk(ctrl_mutex);  // if you already guard controller access
            auto I = std::get<Eigen::VectorXd>(rtc_config.ctrl_LO->get_parameter("integrals"));
            auto E = std::get<Eigen::VectorXd>(rtc_config.ctrl_LO->get_parameter("prev_errors"));
            I.setZero();
            E.setZero();
            rtc_config.ctrl_LO->set_parameter("integrals", I);
            rtc_config.ctrl_LO->set_parameter("prev_errors", E);
        


            updateDMSharedMemory(dm_rtc, rtc_config.zeroCmd);
            ImageStreamIO_sempost(&dm_rtc0, 1);
            std::cout << "[SAFETY] here Zeroing DM due and Reset LO integrals" << std::endl;
            std::cout << "[SAFETY] Re-locking the loop" << std::endl;
            std::cout << " " << std::endl;
        }



        // injecting disturbances in command space 
        // This handles: enabled flag, t_start/t_stop, hold_frames, latency_frames,
        // waveform selection (sine/square/step/chirp/prbs), basis lookup & normalization.
        //compute_c_inj(rtc_config, c_inj);

        //-----8. C L O C K -----//
        // auto start8 = Clock::now();
        
        // If injecting in COMMAND space, build actuator vector AFTER controller outputs
        if (rtc_config.inj_signal.enabled &&
            rtc_config.inj_signal.apply_to == "command")
        {
            const double a = g_inj.last_out_scalar;

            if      (rtc_config.inj_signal.branch == "ALL") c_inj_applied = a * g_inj_cache.p_all;
            else if (rtc_config.inj_signal.branch == "LO")  c_inj_applied = a * g_inj_cache.p_lo;
            else                                            c_inj_applied = a * g_inj_cache.p_ho;
        }

        // auto end8   = Clock::now();
        // auto us8    = std::chrono::duration_cast<std::chrono::nanoseconds>(end8 - start8).count();
        // std::cout << "[TIMER] command space signal injection checks " << us8 << " ns\n";

        //----- E N D  C L O C K -----//

        /// ============= FINAL DM CMD ==============
        dmCmd = -1 * (c_LO + c_HO) + c_inj_applied + ol_lo + ol_ho ; //  ol are open loop offsets distinct from injected signals for system ID (c_inj_applied). ol_* are more usefull for static matrix calibrations for example. See baldr.cpp commander functions relating to interactions.
        /// =========================================

        // remove piston 



        //if (dmCmd.cwiseAbs().maxCoeff() > 0) {
        //    std::cout << "max dmCmd = " << dmCmd.cwiseAbs().maxCoeff() << std::endl;
        //};
        //rtc_config.limits.open_on_dm_limit

        
        if (dmCmd.cwiseAbs().maxCoeff() > 0.4) {
            std::cout << "going bad, stop" << std::endl;
            
            updateDMSharedMemory(dm_rtc, rtc_config.zeroCmd);
            ImageStreamIO_sempost(&dm_rtc0, 1);

            rtc_config.state.take_telemetry=1;
            servo_mode = SERVO_STOP;

            // open_baldr_LO
            // Find index of maximum absolute value.
            //int culprit = 0;
            //double maxVal = u_HO.cwiseAbs().maxCoeff(&culprit);
            //std::cout << "beam " << beam_id << " broke by act." << culprit << ", resetting" << std::endl;
            
            // Optionally, to reduce gain for this actuator by half, you might do:
            // rtc_config.ctrl_HO.ki(culprit) *= 0.5;
            
            // Set the DM command data to zeros.
            // Create a zero vector with the same length as u_HO.
            
            // Convert the vector to a 2D map via dm.cmd_2_map2D() and write to shared memory.
            //dm.set_data(dm.cmd_2_map2D(zeroCmd));

            // // zero dm
            // //updateDMSharedMemory( zeroCmd ) ;
            
            // // Reset the high-order controller.
            // rtc_config.ctrl_HO.reset();
            // rtc_config.ctrl_LO.reset();
            
            // // Increase the counter for this actuator.
            // naughty_list[culprit] += 1;
            
            // // If the controller's gain vector has fewer than 30 elements, stop the loop.
            // if (naughty_list[culprit] > 10) {
            //     servo_mode = SERVO_STOP;
            //     std::cout << "ENDING" << std::endl;
            // }
            
            // // If this actuator has been problematic more than 4 times, disable it by setting its gain to zero.
            // if (naughty_list[culprit] > 4) {
            //     std::cout << "Turn off naughty actuator " << culprit << std::endl;
            //     rtc_config.ctrl_HO.ki(culprit) = 0.0;
            // }
        }


        // ******************** UPDATE DM ******************************

        //-----9. C L O C K -----//
        // auto start9 = Clock::now();

        updateDMSharedMemory(dm_rtc, dmCmd);

        // Signal the master DM process to update itself.
        ImageStreamIO_sempost(&dm_rtc0, 1);

        // auto end9   = Clock::now();
        // auto us9    = std::chrono::duration_cast<std::chrono::nanoseconds>(end9 - start9).count();
        // std::cout << "[TIMER] update DM SHM and post sem " << us9 << " ns\n";

        //BCB
        //updateDMSharedMemory( dmCmd ) ;

        // ******************** --------- ******************************

        // ************************************************************
        // Struc of ring buffers to keep history and offload to telemetry thread if requestec 
        if (true){

            //-----10. C L O C K -----//
            auto start10 = Clock::now();

            
            std::lock_guard<std::mutex> lock(telemetry_mutex);
            // Get the current time as a double (microseconds)
            current_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch()
                                    ).count();
                                    
            // Append data to telemetry ring buffers.
            //std::cout << "time  " << current_time_ms  << std::endl;
            rtc_config.telem.timestamp.push_back(current_time_ms);

            rtc_config.telem.LO_servo_mode.push_back(servo_mode_LO);          // 'c_HO' is Eigen::VectorXd

            rtc_config.telem.HO_servo_mode.push_back(servo_mode_HO);          // 'c_HO' is Eigen::VectorXd

            //std::cout << "img size: " << img.size() << std::endl;
            rtc_config.telem.img.push_back(img);     // we do the RAW img here, not the reduced dark subtracted image     // 'img' is Eigen::VectorXd
            //std::cout << "img_dm size: " << img_dm.size() << std::endl;
            rtc_config.telem.img_dm.push_back(img_dm);          // 'img' is Eigen::VectorXd
            //std::cout << ", sig size: " << sig.size() << std::endl;
            
            /// WE DO THIS WHEN WE ARE IN LOOP TO CALCULATE AVERAGE
            //rtc_config.telem.signal.push_back(sig);       // 'sig' is Eigen::VectorXd

            //std::cout << ", sig size: " << sig.size() << std::endl;
            rtc_config.telem.e_LO.push_back(e_LO);          // 'e_LO' is Eigen::VectorXd

            //std::cout << ", sig size: " << sig.size() << std::endl;
            rtc_config.telem.u_LO.push_back(u_LO);          // 'u_LO' is Eigen::VectorXd

            //std::cout << ", sig size: " << sig.size() << std::endl;
            rtc_config.telem.e_HO.push_back(e_HO);          // 'e_HO' is Eigen::VectorXd

            //std::cout << ", sig size: " << sig.size() << std::endl;
            rtc_config.telem.u_HO.push_back(u_HO);          // 'u_HO' is Eigen::VectorXd

            //std::cout << ", sig size: " << c_LO.size() << std::endl;
            rtc_config.telem.c_LO.push_back(c_LO);          // 'c_LO' is Eigen::VectorXd

            //std::cout << ", c_HO size: " << c_HO.size() << std::endl;
            rtc_config.telem.c_HO.push_back(c_HO);          // 'c_HO' is Eigen::VectorXd

            // just first iteration to test save fits function update! 
            rtc_config.telem.c_inj.push_back(c_inj_applied); // <-- NEW
            
            rtc_config.telem.rmse_est.push_back(dm_rms_est_1);          // New 
     

            // === SNR over full img_dm telemetry buffer (recomputed each frame) ===
            // Definition: < mean_t(img_dm) / std_t(img_dm) >_space over all frames currently stored.
            // doing dumb slower implementation of recalculating every iteration, but simple
            double snr_value = 0.0;

            const auto& buf = rtc_config.telem.img_dm;
            const size_t T = buf.size();

            if (T >= 2) {
                // Use the last frame's size as the reference spatial size
                const Eigen::Index P = buf.back().size();

                Eigen::VectorXd sum   = Eigen::VectorXd::Zero(P);
                Eigen::VectorXd sumsq = Eigen::VectorXd::Zero(P);

                size_t used_frames = 0;
                for (const auto& x : buf) {
                    if (x.size() != P) continue;                   // skip anomalous frames
                    sum   += x;
                    sumsq += x.cwiseProduct(x);
                    ++used_frames;
                }

                if (used_frames >= 2) {
                    const double n = static_cast<double>(used_frames);
                    Eigen::VectorXd mean = sum / n;

                    // sample variance per pixel: (sumx^2 - n*mu^2)/(n-1)
                    Eigen::VectorXd var_num = sumsq - n * mean.cwiseProduct(mean);
                    var_num = var_num.cwiseMax(0.0);               // numeric guard
                    Eigen::ArrayXd  stdev   = (var_num / (n - 1.0)).array().sqrt();

                    // Spatial average of per-pixel mean/std; skip zero-std pixels
                    double acc = 0.0;
                    size_t used_px = 0;
                    for (Eigen::Index i = 0; i < P; ++i) {
                        double s = stdev(i);
                        if (s > 0.0) {
                            acc += mean(i) / s;
                            ++used_px;
                        }
                    }
                    snr_value = (used_px > 0) ? (acc / static_cast<double>(used_px)) : 0.0;
                }
            }

            // Store to telemetry
            rtc_config.telem.snr.push_back(snr_value);

            // // === SNR from img_dm telemetry buffer ===
            // // Definition: < mean_t(img_dm) / std_t(img_dm) >_space

            // // Persistent per-pixel accumulators (reset on first use or size change)
            // static Eigen::VectorXd snr_mean;   // per-pixel running mean
            // static Eigen::VectorXd snr_M2;     // per-pixel sum of squared diffs
            // static size_t snr_count = 0;

            // // (Re)initialise if needed
            // if (snr_count == 0 || snr_mean.size() != img_dm.size()) {
            //     snr_mean  = Eigen::VectorXd::Zero(img_dm.size());
            //     snr_M2    = Eigen::VectorXd::Zero(img_dm.size());
            //     snr_count = 0;
            // }

            // // Welford update with current img_dm sample (vectorised, per-pixel) - an unbiased estimate
            // snr_count += 1;
            // Eigen::VectorXd delta  = img_dm - snr_mean;
            // snr_mean.noalias()    += delta / static_cast<double>(snr_count);
            // Eigen::VectorXd delta2 = img_dm - snr_mean;
            // snr_M2.noalias()      += delta.cwiseProduct(delta2);

            // double snr_value = 0.0;

            // if (snr_count > 1) {
            //     // sample variance/std per pixel
            //     Eigen::VectorXd var = snr_M2 / static_cast<double>(snr_count - 1);
            //     var = var.cwiseMax(0.0);                     // numerical guard
            //     Eigen::ArrayXd std = var.array().sqrt();

            //     // ratio per pixel; guard zero-std (skip those pixels)
            //     double sum_ratio = 0.0;
            //     size_t n_used    = 0;
            //     for (Eigen::Index i = 0; i < std.size(); ++i) {
            //         double s = std(i);
            //         if (s > 0.0 && std::isfinite(snr_mean(i))) {
            //             sum_ratio += snr_mean(i) / s;
            //             ++n_used;
            //         }
            //     }
            //     snr_value = (n_used > 0) ? (sum_ratio / static_cast<double>(n_used)) : 0.0;
            // } else {
            //     // fewer than 2 frames: std undefined -> report 0 for now
            //     snr_value = 0.0;
            // }

            // add to telemetry
            //rtc_config.telem.snr.push_back(snr_value);

            //// ------------ older method 
            // double sum = 0.0;
            // double sumsq = 0.0;
            // size_t count = 0;


            // for (Eigen::Index i = 0; i < img_dm.size(); ++i) {
                
            //     double val = img_dm(i);
            //     sum += val;
            //     sumsq += val * val;
            //     ++count;
                
            // }

            // double snr_value = 0.0;
            // if (count > 0) {
            //     double mean = sum / count;
            //     double variance = (sumsq / count) - (mean * mean);
            //     double stddev = (variance > 0.0) ? std::sqrt(variance) : 0.0;
            //     snr_value = (stddev != 0.0) ? (mean / stddev) : 0.0;
            // }

            // for (Eigen::Index i = 0; i < sig.size(); ++i) {
            //     if (rtc_config.filters.inner_pupil_filt(i) > 0.5) {
            //         double val = sig(i);
            //         sum += val;
            //         sumsq += val * val;
            //         ++count;
            //     }
            // }

            // double snr_value = 0.0;
            // if (count > 0) {
            //     double mean = sum / count;
            //     double variance = (sumsq / count) - (mean * mean);
            //     double stddev = (variance > 0.0) ? std::sqrt(variance) : 0.0;
            //     snr_value = (stddev != 0.0) ? (mean / stddev) : 0.0;
            // }

            // Add to telemetry
            //rtc_config.telem.snr.push_back(snr_value);

            // Increment the counter.

            rtc_config.telem.counter++;

            // auto end10   = Clock::now();
            // auto us10    = std::chrono::duration_cast<std::chrono::nanoseconds>(end10 - start10).count();
            // std::cout << "[TIMER] update telemetry ring buffer " << us10 << " ns\n";


        }

        //// getting telemetry in AIV 
        // // write telemetry to some shared memory , latecny of this NOT tested. TBD if we keep this method
        // if (true){

        //     std::lock_guard<std::mutex> lock(telemetry_mutex);

        //     //int shm_idx = shm_telem_cnt % shm_telem_samples;

        //     // Get write pointers
        //     float* buf_sig = (float*) shm_sig.array.F;
        //     float* buf_eLO = (float*) shm_eLO.array.F;
        //     float* buf_eHO = (float*) shm_eHO.array.F;

        //     // Compute offset for this frame
        //     size_t offset_sig = 140 * (shm_telem_cnt % shm_telem_samples);
        //     size_t offset_eLO = 2 * (shm_telem_cnt % shm_telem_samples);
        //     size_t offset_eHO = 140 * (shm_telem_cnt % shm_telem_samples);

        //     // Write current frame
        //     memcpy(&buf_sig[offset_sig], sig.data(), 140 * sizeof(float));
        //     memcpy(&buf_eLO[offset_eLO], e_LO.data(), 2 * sizeof(float));
        //     memcpy(&buf_eHO[offset_eHO], e_HO.data(), 140 * sizeof(float));

        //     // Update metadata
        //     shm_sig.md->cnt0++;
        //     shm_sig.md->cnt1++;
        //     shm_eLO.md->cnt0++;
        //     shm_eLO.md->cnt1++;
        //     shm_eHO.md->cnt0++;
        //     shm_eHO.md->cnt1++;

        //     // Increment index
        //     shm_telem_cnt++;
        // }

        // -------------------- DEAD TIME BEGINS HERE 
        // schedule next wakeup
        // next_tick += loop_time;
        // std::this_thread::sleep_until(next_tick);

        // detect overrun
        // auto now = std::chrono::steady_clock::now();
        // if (now > next_tick) {
        //     auto over = now - next_tick;
        //     //std::cerr<<"Loop overran by "
        //     //       << std::chrono::duration_cast<std::chrono::microseconds>(over).count()
        //     //       <<" us\n";
        // }
        
        //end = std::chrono::steady_clock::now();
        //duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // with telemetry and everything typically 220us loop without sleep (5kHz)
        //std::cout << "duration microsec:  " << duration.count() << std::endl;

        /////////////////////// SLEEP ///////////////////////

        //std::cout << servo_mode_LO.load() << std::endl;
        
        // if (duration < loop_time){
        //    std::this_thread::sleep_for(std::chrono::microseconds(loop_time - duration));
        // }


    }
    std::cout << "servo_mode changed to" << servo_mode << "...rtc stopped" << std::endl;
}












