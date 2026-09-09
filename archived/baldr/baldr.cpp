#define TOML_IMPLEMENTATION
#include "baldr.h"
#include "burst_window.h"
#include <commander/commander.h>
#include <math.h> // For old-style C mathematics if needed.
#include <zmq.hpp>
#include <fitsio.h>
#include <string>
#include <regex>
#include <unordered_set>
#include <cctype>   // (for std::tolower used in update_ctrl_parameter)
#include <filesystem> //C++ 17

// Commander struct definitions for json. This is in a separate file to keep the main code clean.
#include "commander_structs.h"
#include <atomic>
#include <condition_variable>


#include <unordered_map>
#include <functional>
#include <type_traits>


using json = nlohmann::json;
//----------Globals-------------------

// The input configuration
// this is required inpy to the baldr RTC and gets set by user choice in main 

int beam_id; 
std::string phasemask;
toml::table config;
bdr_rtc_config rtc_config;

std::string observing_mode = "bright"; // bright - 12x12 (default) or faint - 6x6 pixels
std::string user_config_filename = "";
bool user_provided_config = false;      // true if user gives a manual toml file



// loop time , not nessary when using semaphores for frames - but can be useful for testing 
//float loop_time = 0.0f; // set later
//bool loop_time_override = false;

// int servo_mode;
std::atomic<int> servo_mode; 
std::atomic<int> servo_mode_LO;
std::atomic<int> servo_mode_HO;


/////// new stuff for onsky interactions (19/8/25)
// Definition of the extern declared in baldr.h 
// to hold baldr (DM channel 2) open loop offsets 
std::shared_ptr<const OLOffsets> g_ol_offsets{nullptr};

double g_subframe_int = 1.0; // sum of intensity in the current subframe //post TTonsky

std::string telemFormat = "fits";//"json";
std::string telem_save_path = "/home/asg/Music/"; // /home/rtc/Downloads/"; //"/home/benjamin/Downloads/"//"/home/asg/Music/";

// Your basis file location
static constexpr const char* kDMBaseFITS =
    "/usr/local/etc/baldr/dm_basis/bmc_multi3p5_dm_bases.fits";



// Servo parameters. These are the parameters that will be adjusted by the commander

// Mutex for the RTC.
std::mutex rtc_mutex;
// mutex telemetry
std::mutex telemetry_mutex;
// mutex for controller gain updates:
std::mutex ctrl_mutex;
//std::mutex ctrl_LO_mutex;//later
//std::mutex ctrl_HO_mutex;
std::atomic<bool> pause_rtc(false);           // When true, RTC loop should pause. atomic to be thread safe 
std::mutex rtc_pause_mutex;  //protect access to shared state (in this case, the pause flag and any associated state) while waiting and notifying.
std::condition_variable rtc_pause_cv; //condition variable is used to block (put to sleep) the RTC thread until a particular condition is met

//// uncomment and build July 2025 AIV
std::atomic<int> global_boxcar{1};  // for possible weighted averaging of signals in rtc

IMAGE subarray; // The C-red subarray
IMAGE dm_rtc; // The DM subarray
IMAGE dm_rtc0; // The DM subarray master to post semaphores to


// foprward dec.
void pauseRTC();
void resumeRTC();

// Initialize global ZMQ variables
zmq::context_t cam_zmq_context(1);
zmq::socket_t cam_zmq_socket(cam_zmq_context, zmq::socket_type::req);
std::string cam_host_str = "tcp://192.168.100.2:6667";
bool cam_zmq_initialized = false;

// Initialize global ZMQ variables for MultiDeviceServer
zmq::context_t mds_zmq_context(1);
zmq::socket_t mds_zmq_socket(mds_zmq_context, zmq::socket_type::req);
std::string mds_host_str = "tcp://192.168.100.2:5555";
bool mds_zmq_initialized = false;


std::atomic<uint64_t> g_inj_cfg_epoch{0};

// Tracks the currently loaded Baldr TOML config file (full path)
std::string g_active_config_filename;

// Forward declaration (implemented in telemetry.cpp)
std::vector<std::string> list_all_numeric_telem_fields();

// From telemetry.cpp:
// bool write_fields_to_fits(const std::string& path,
//                           const std::vector<std::string>& fields,
//                           std::function<bool(std::string_view, Eigen::MatrixXd&, std::string&)> extract,
//                           long nsteps);

////updated
bool write_fields_to_fits(const std::string& path,
                          const std::vector<std::string>& fields,
                          std::function<bool(std::string_view, Eigen::MatrixXd&, std::string&)> extract,
                          long nsteps,
                          const std::vector<std::pair<std::string, std::string>>& header_strs = {},
                          const std::vector<std::pair<std::string, long>>&    header_longs = {},
                          const std::vector<std::pair<std::string, double>>&  header_doubles = {});

bool telem_extract_matrix_lastN(const bdr_telem& telem,
                                std::mutex& telemetry_mutex,
                                std::string_view field,
                                std::size_t N,
                                Eigen::MatrixXd& M,
                                std::string& why);


// 20-8-25
std::unique_ptr<Controller>
make_controller(const std::string& type, const bdr_controller& cfg) {

    const int n = static_cast<int>(cfg.set_point.size());
 
    if (type == "PID") {
        return std::make_unique<PIDController_1>(
            cfg.kp, cfg.ki, cfg.kd,
            cfg.lower_limits, cfg.upper_limits,
            cfg.set_point
        );
    }
    // extend this mapping later:
    // else if (type == "LEAKY")   return std::make_unique<LeakyIntegratorController>(cfg.kp /*K*/, cfg.ki /*alpha*/, /*dt=*/1.0);
    // else if (type == "KALMAN")  return std::make_unique<KalmanController>(/* sizes */, /* dt */);
    else if (type == "LEAKY") {
        // Map: K <- cfg.kp, alpha <- cfg.ki, dt <- 1/fps (fallback 1.0)
        Eigen::VectorXd K = cfg.kp;

        Eigen::VectorXd alpha;
        if (cfg.ki.size() == n) {
            alpha = cfg.ki;
        } else {
            // default alpha if not provided: moderately leaky
            alpha = Eigen::VectorXd::Constant(n, 0.95);
        }
        // clamp alpha into [0,1)
        for (Eigen::Index i=0;i<alpha.size();++i)
            alpha(i) = std::min(std::max(alpha(i), 0.0), 0.999999);

        // dt from current RTC fps if available
        const double dt = (rtc_config.fps > 1e-9) ? (1.0 / rtc_config.fps) : 1.0;

        auto c = std::make_unique<LeakyIntegratorController>(K, alpha, dt);
        // optional: seed setpoint if sized
        if (cfg.set_point.size()==n) c->set_setpoint(cfg.set_point);
        return c;
    }
    else if (type == "KALMAN") {
        // Minimal constructor: choose sizes and dt sensibly.
        // State dimension ~ number of actuators/channels (use kp size).
        const int nx = n;

        // Measurement dimension: if you have a known measurement size, use it.
        // Fall back to nx if unknown.
        int nz = nx;
        // If your config carries a measurement size somewhere, substitute it here.

        const double dt = (rtc_config.fps > 1e-9) ? (1.0 / rtc_config.fps) : 1.0;

        auto c = std::make_unique<KalmanController>(nx, nz, dt);

        // optional: seed setpoint to match PID/LEAKY convention
        if (cfg.set_point.size()==nx) c->set_setpoint(cfg.set_point);

        // NOTE: If you want non-default A,B,H,Q,R,P, set them via c->set_parameter(...) after construction.
        // (e.g., names: "A","B","H","Q","R","P" depending on your KalmanController implementation).

        return c;
    }

    throw std::runtime_error("Unknown controller_type: " + type);
}

// to make changes to injection config visible to rtc thread
void mark_injection_changed() {
    g_inj_cfg_epoch.fetch_add(1, std::memory_order_relaxed);
}


// // Parameterized constructor.
// PIDController::PIDController(const Eigen::VectorXd& kp_in,
//                              const Eigen::VectorXd& ki_in,
//                              const Eigen::VectorXd& kd_in,
//                              const Eigen::VectorXd& lower_limit_in,
//                              const Eigen::VectorXd& upper_limit_in,
//                              const Eigen::VectorXd& setpoint_in)
//     : kp(kp_in),
//       ki(ki_in),
//       kd(kd_in),
//       lower_limits(lower_limit_in),
//       upper_limits(upper_limit_in),
//       set_point(setpoint_in),
//       ctrl_type("PID")
// {
//     int size = kp.size();
//     if (ki.size() != size || kd.size() != size ||
//         lower_limits.size() != size || upper_limits.size() != size ||
//         set_point.size() != size) {
//         throw std::invalid_argument("All input vectors must have the same size.");
//     }
//     output = Eigen::VectorXd::Zero(size);
//     integrals = Eigen::VectorXd::Zero(size);
//     prev_errors = Eigen::VectorXd::Zero(size);
// }

// // constructor that accepts a bdr_controller struct.
// PIDController::PIDController(const bdr_controller& config_in)
//     : PIDController(config_in.kp, config_in.ki, config_in.kd, config_in.lower_limits, config_in.upper_limits, config_in.set_point)
// {
// }

// // Default constructor.
// PIDController::PIDController()
//     : kp(Eigen::VectorXd::Zero(140)),
//       ki(Eigen::VectorXd::Zero(140)),
//       kd(Eigen::VectorXd::Zero(140)),
//       lower_limits(-Eigen::VectorXd::Ones(140)),
//       upper_limits(Eigen::VectorXd::Ones(140)),
//       set_point(Eigen::VectorXd::Zero(140)),
//       ctrl_type("PID")
// {
//     int size = kp.size();
//     output = Eigen::VectorXd::Zero(size);
//     integrals = Eigen::VectorXd::Zero(size);
//     prev_errors = Eigen::VectorXd::Zero(size);
// }

// Eigen::VectorXd PIDController::process(const Eigen::VectorXd& measured) {
//     int size = set_point.size();
//     if (measured.size() != size) {
//         throw std::invalid_argument("Input vector size must match setpoint size.");
//     }
//     if (kp.size() != size || ki.size() != size || kd.size() != size ||
//         lower_limits.size() != size || upper_limits.size() != size) {
//         throw std::invalid_argument("Input vectors of incorrect size.");
//     }
//     if (integrals.size() != size) {
//         std::cout << "Reinitializing integrals, prev_errors, and output to zero with correct size.\n";
//         integrals = Eigen::VectorXd::Zero(size);
//         prev_errors = Eigen::VectorXd::Zero(size);
//         output = Eigen::VectorXd::Zero(size);
//     }
//     for (int i = 0; i < size; ++i) {
//         double error = measured(i) - set_point(i);
//         if (ki(i) != 0.0) {
//             integrals(i) += error;
//             if (integrals(i) < lower_limits(i))
//                 integrals(i) = lower_limits(i);
//             if (integrals(i) > upper_limits(i))
//                 integrals(i) = upper_limits(i);
//         }
//         double derivative = error - prev_errors(i); // if error bigger than previous error you want to dampen output
//         output(i) = kp(i) * error + ki(i) * integrals(i) + kd(i) * derivative;
//         prev_errors(i) = error;
//     }
//     return output;
// }

// void PIDController::set_all_gains_to_zero() {
//     kp = Eigen::VectorXd::Zero(kp.size());
//     ki = Eigen::VectorXd::Zero(ki.size());
//     kd = Eigen::VectorXd::Zero(kd.size());
// }

// void PIDController::reset() {
//     integrals.setZero();
//     prev_errors.setZero();
//     output.setZero();
// }


//----------helper functions from here---------------

// Function to find the most recent config file for a given beam and mode (default = "bright")
std::string find_latest_config_file(int beam_id, const std::string& mode = "bright") {
    const std::string base_dir = "/usr/local/etc/baldr/rtc_config/";
    const std::string filename_prefix = "baldr_config_" + std::to_string(beam_id) + "_" + mode + "_";
    const std::string filename_suffix = ".toml";

    std::vector<std::filesystem::directory_entry> matching_files;

    for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(base_dir)) {
        if (dir_entry.is_regular_file()) {
            const std::string filename = dir_entry.path().filename().string();
            if (filename.size() >= filename_prefix.size() + filename_suffix.size() &&
                filename.substr(0, filename_prefix.size()) == filename_prefix &&
                filename.substr(filename.size() - filename_suffix.size()) == filename_suffix) {
                matching_files.push_back(dir_entry);
            }
        }
    }

    if (matching_files.empty()) {
        throw std::runtime_error("No config file found for beam " + std::to_string(beam_id) + " with mode '" + mode + "'.");
    }

    // Sort newest first based on filename (descending)
    std::sort(matching_files.begin(), matching_files.end(),
        [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) {
            return a.path().filename().string() > b.path().filename().string();
        }
    );

    return matching_files.front().path().string();
}


/// get status for MCS to send to wag DB 

Status get_status() {
    Status status;

    // Loop states from atomics
    status.TT_state = servo_mode_LO.load();
    status.HO_state = servo_mode_HO.load();

    // Mode / mask straight from globals
    status.mode      = observing_mode.empty()
                         ? std::string("unknown")
                         : observing_mode;   // "bright" or "faint"
    status.phasemask = phasemask.empty()
                         ? std::string("unknown")
                            : phasemask;        // e.g., "H3"
    // Frequency: use runtime value set in initDerivedParameters()
    // (This is the *actual* camera/RTC rate read via ZMQ, not the string in cam.fps)
    status.frequency = rtc_config.fps;   // double

    // Configured flag: consider “configured” if we have a non-empty runtime I2M matrix
    const bool lo_ready = (rtc_config.matrices.I2M_LO.size() > 0);
    const bool ho_ready = (rtc_config.matrices.I2M_HO.size() > 0);
    status.configured   = (lo_ready || ho_ready) ? 1 : 0;

    // Controller type (PID/Leaky/Kalman) from state
    status.ctrl_type = rtc_config.state.controller_type;

    // Active config file path
    status.config_file = g_active_config_filename.empty()
                         ? std::string("unknown")
                         : g_active_config_filename;

    // Signal injection master enable
    status.inj_enabled = rtc_config.inj_signal.enabled ? 1 : 0;

    // Auto-loop: expose a single bit if either auto_close or auto_open is set
    status.auto_loop = rtc_config.state.auto_close  ? 1 : 0;

    // Thresholds from bdr_limits (strehl-based open/close)
    status.close_on_snr = 2.0; // should be a float
    status.open_on_snr = rtc_config.limits.open_on_flux_limit; // should be a float
    status.close_on_strehl = rtc_config.limits.close_on_strehl_limit;    
    status.open_on_strehl = rtc_config.limits.open_on_strehl_limit;
    status.TT_offsets = 0; // 1 or 0 

    // If your Status has other optional fields (e.g., Strehl/SNR estimates),
    // you can surface last-computed values here when available.

    return status;


    /** baldr fields
    TT_state: int
    HO_state: int
    mode: str
    phasemask: str
    frequency: float
    configured: int
    ctrl_type: str
    // -- --- complete: bool -- this isnt set here
    config_file: str
    inj_enabled: int
    auto_loop: int
    close_on_snr: float
    open_on_snr: float
    close_on_strehl: float
    open_on_strehl: float
    TT_offsets: int
    /// --- x_pup_offset: float -- this isnt done by rtc
    /// --- y_pup_offset: float -- this isnt done by rtc **/



}


// // Function to find the most recent config file for a given beam
// std::string find_latest_config_file(int beam_id) {
//     const std::string base_dir = "/usr/local/etc/baldr/rtc_config/";
//     const std::string filename_prefix = "baldr_config_" + std::to_string(beam_id) + "_";
//     const std::string filename_suffix = ".toml";

//     std::vector<std::filesystem::directory_entry> matching_files;

//     for (const auto& dir_entry : std::filesystem::recursive_directory_iterator(base_dir)) {
//         if (dir_entry.is_regular_file()) {
//             const std::string filename = dir_entry.path().filename().string();
//             if (filename.size() >= filename_prefix.size() + filename_suffix.size() &&
//                 filename.substr(0, filename_prefix.size()) == filename_prefix &&
//                 filename.substr(filename.size() - filename_suffix.size()) == filename_suffix) {
//                 matching_files.push_back(dir_entry);
//             }
//         }
//     }

//     if (matching_files.empty()) {
//         throw std::runtime_error("No config file found for beam " + std::to_string(beam_id));
//     }

//     // Sort newest first based on filename (descending)
//     std::sort(matching_files.begin(), matching_files.end(),
//         [](const std::filesystem::directory_entry& a, const std::filesystem::directory_entry& b) {
//             return a.path().filename().string() > b.path().filename().string();
//         }
//     );

//     return matching_files.front().path().string();
// }


// Asgard Camera Server
void init_cam_zmq() {
    if (!cam_zmq_initialized) {
        cam_zmq_socket.connect(cam_host_str);
        cam_zmq_initialized = true;
    }
}

std::string send_cam_cmd(const std::string& command) {
    init_cam_zmq();
    std::string full_cmd = "cli \"" + command + "\"";
    cam_zmq_socket.send(zmq::buffer(full_cmd), zmq::send_flags::none);
    zmq::message_t reply;
    cam_zmq_socket.recv(reply, zmq::recv_flags::none);
    return std::string(static_cast<char*>(reply.data()), reply.size());
}


json send_cam_command(json args) {
    if (!args.is_array() || args.size() != 1) {
        return json{{"error", "Expected a single string argument [\"command\"]"}};
    }
    try {
        std::string command = args.at(0).get<std::string>();

        std::string response = send_cam_cmd(command);
        return json{{"command_sent", command}, {"camera_response", response}};
    } catch (const std::exception& ex) {
        return json{{"error", ex.what()}};
    }
}

std::string extract_value(const std::string& response) {
    std::regex re("^\"\\s*(.*?)\\\\r\\\\n");
    std::smatch match;
    if (std::regex_search(response, match, re)) {
        return match[1];
    }
    return "N/A";
}

float get_float_cam_param(const std::string& command) {
    std::string response = send_cam_cmd(command);
    std::string val_str = extract_value(response);
    try {
        return std::stof(val_str);
    } catch (...) {
        std::cerr << "Failed to convert camera response to float: " << val_str << std::endl;
        return -1.0f;
    }
}


// MDS 

void init_mds_zmq() {
    if (!mds_zmq_initialized) {
        mds_zmq_socket.connect(mds_host_str);
        mds_zmq_initialized = true;
    }
}

std::string send_mds_cmd(const std::string& message) {
    init_mds_zmq();
    mds_zmq_socket.send(zmq::buffer(message), zmq::send_flags::none);
    zmq::message_t reply;
    auto result = mds_zmq_socket.recv(reply, zmq::recv_flags::none);
    if (!result.has_value()) {
        throw std::runtime_error("Timeout or error receiving reply from MDS.");
    }
    return std::string(static_cast<char*>(reply.data()), reply.size());
}

json send_mds_command(json args) {
    if (!args.is_array() || args.size() != 1) {
        return json{{"error", "Expected a single string argument [\"command\"]"}};
    }
    try {
        std::string command = args.at(0).get<std::string>();

        std::string response = send_mds_cmd(command);
        return json{{"command_sent", command}, {"mds_response", response}};
    } catch (const std::exception& ex) {
        return json{{"error", ex.what()}};
    }
}

// Helper function to split a comma-separated string into a vector of ints.
std::vector<int> split_indices(const std::string &indices_str) {
    std::vector<int> indices;
    std::stringstream ss(indices_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) {
            try {
                indices.push_back(std::stoi(token));
            } catch (const std::exception &ex) {
                std::cerr << "Error converting token '" << token << "': " << ex.what() << std::endl;
            }
        }
    }
    return indices;
}


// Configure burst window (used for intensity slope estime) from current camera settings
//burst update here
void configure_burstWindow_from_camera(double phase = 0.0) {
    std::lock_guard<std::mutex> lock(rtc_mutex);

    try {
        // Query number of reads per reset and convert to integer
        std::string nbread_str = extract_value(send_cam_cmd("nbreadworeset"));
        int nbread = std::stoi(nbread_str);

        // Query current FPS
        float fps = get_float_cam_param("fps raw");

        // Sanity check
        if (nbread <= 0 || fps <= 0.0) {
            throw std::runtime_error("Invalid nbread or fps from camera");
        }

        // Configure the burst window
        rtc_config.burst.configure(nbread, fps, phase);


    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to configure burst window from camera: " << e.what() << std::endl;
    }
}
//commander compatible wrapper 
void cmd_configure_burst(json args) {
    double phase = 0.0;
    if (args.is_array() && !args.empty()) {
        phase = args.at(0).get<double>();
    }

    try {
        configure_burstWindow_from_camera(phase);
;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] failed to configure burst window " << e.what() << std::endl;
    }
}



toml::table readConfig(const std::string &filename) {
    try {
        // Parse the TOML file and return the table.
        return toml::parse_file(filename);
    }
    catch(const toml::parse_error &err) {
        std::cerr << "TOML parse error in file " << filename << ": " 
                  << err.description() << "\n";
        std::exit(1);
    }
}


// Helper function to read RTC config from a TOML table (from the baldr calibration pipeline). const std::string &filename,
bdr_rtc_config readBDRConfig(const toml::table& config, const std::string& beamKey, const std::string& phaseKey) {
    
    bdr_rtc_config rtc;
    
    // Read the TOML file.
    //toml::table tmpconfig = readConfig(filename);
    
    // bdr_reduction: assume these keys exist in the beam-specific ctrl_model.
    auto beam_node = config[beamKey];
    if (!beam_node || !beam_node.is_table()) {
        std::cerr << "Beam configuration not found for key: " << beamKey << std::endl;
        std::exit(1);
    }
    auto beam_tbl = *beam_node.as_table();
    // Check if the specified phase mask exists.
    if (!beam_tbl.contains(phaseKey)) {
        std::cerr << "Error: Phase mask \"" << phaseKey
                  << "\" not found in configuration for beam " << beamKey << std::endl;
        throw std::runtime_error("Phase mask not found");
    }
    {
        auto phase_tbl = *beam_tbl[phaseKey].as_table();
        
        if (auto ctrl_node = phase_tbl["ctrl_model"]; ctrl_node && ctrl_node.is_table()) {
            auto ctrl_tbl = *ctrl_node.as_table();

            // state 
            try{ 
                rtc.state.DM_flat = ctrl_tbl["DM_flat"] ? ctrl_tbl["DM_flat"].value_or(std::string("")) : "";
                rtc.state.signal_space  =  ctrl_tbl["signal_space"] ? ctrl_tbl["signal_space"].value_or(std::string("")) : ""; 
                rtc.state.LO = ctrl_tbl["LO"] ? ctrl_tbl["LO"].value_or(0) : 0;
                rtc.state.controller_type = ctrl_tbl["controller_type"] ? ctrl_tbl["controller_type"].value_or(std::string("")) : "";
                rtc.state.inverse_method_LO = ctrl_tbl["inverse_method_LO"] ? ctrl_tbl["inverse_method_LO"].value_or(std::string("")) : "";
                rtc.state.inverse_method_HO = ctrl_tbl["inverse_method_HO"] ? ctrl_tbl["inverse_method_HO"].value_or(std::string("")) : "";
                rtc.state.auto_close = ctrl_tbl["auto_close"] ? ctrl_tbl["auto_close"].value_or(int(0)) : 0;
                rtc.state.auto_open = ctrl_tbl["auto_open"] ? ctrl_tbl["auto_open"].value_or(int(1)): 1;
                rtc.state.auto_tune = ctrl_tbl["auto_tune"] ? ctrl_tbl["auto_tuen"].value_or(int(0)) : 0;
                rtc.state.simulation_mode = 0; 
            }catch (const std::exception& ex) {
                std::cerr << "Error with state read-in: " << ex.what() << std::endl;
                std::exit(1);
            }
            // pix
            // reduction products 
            try{ 
                rtc.reduction.bias = convertTomlArrayToEigenMatrix(*ctrl_tbl["bias"].as_array(),rtc.reduction.bias,"bias");
                rtc.reduction.dark = convertTomlArrayToEigenMatrix(*ctrl_tbl["dark"].as_array(),rtc.reduction.dark,"dark");
                // For bias_dm and dark_dm, these are computed later from I2A.
            }catch (const std::exception& ex) {
                std::cerr << "Error processing rtc.reduction: " << ex.what() << std::endl;
                std::exit(1);
            }
            // pixels 
            try{ 
                rtc.pixels.crop_pixels = convertTomlArrayToEigenMatrix(*ctrl_tbl["crop_pixels"].as_array(),rtc.pixels.crop_pixels,"crop_pixels"); // r1,r2,c1,c2 cropped pupil coordinates in global frame
                rtc.pixels.pupil_pixels = convertTomlArrayToEigenMatrix(*ctrl_tbl["pupil_pixels"].as_array(),rtc.pixels.pupil_pixels,"pupil_pixels"); // pupil pixels (local cropped frame)
                rtc.pixels.bad_pixels = convertTomlArrayToEigenMatrix(*ctrl_tbl["bad_pixels"].as_array(),rtc.pixels.bad_pixels,"bad_pixels"); // bad pixels (local cropped frame)
                rtc.pixels.interior_pixels = convertTomlArrayToEigenMatrix(*ctrl_tbl["interior_pixels"].as_array(),rtc.pixels.interior_pixels,"interior_pixels"); // strict interior (no boundary) pupil pixels (local cropped frame)
                rtc.pixels.secondary_pixels = convertTomlArrayToEigenMatrix(*ctrl_tbl["secondary_pixels"].as_array(),rtc.pixels.secondary_pixels,"secondary_pixels"); // secondary obstrcution shaddow pixels (local cropped frame)
                rtc.pixels.exterior_pixels = convertTomlArrayToEigenMatrix(*ctrl_tbl["exterior_pixels"].as_array(),rtc.pixels.exterior_pixels,"exterior_pixels"); // pixels exterior to pupil with high diffraction from phasemask (local cropped frame)
            }catch (const std::exception& ex) {
                std::cerr << "Error processing rtc.pixels: " << ex.what() << std::endl;
                std::exit(1);
            }

            //Filters, somewhat redundant in pixel space with the pixels struct - but could be used optimally in some cases, especially when projecting to DM space to apply weighting for boundaries!
            try{ 
                rtc.filters.bad_pixel_mask = convertTomlArrayToEigenMatrix(*ctrl_tbl["bad_pixel_mask"].as_array(),rtc.filters.bad_pixel_mask,"bad_pixel_mask"); 
                rtc.filters.pupil = convertTomlArrayToEigenMatrix(*ctrl_tbl["pupil"].as_array(),rtc.filters.pupil,"pupil"); 
                rtc.filters.secondary = convertTomlArrayToEigenMatrix(*ctrl_tbl["secondary"].as_array(),rtc.filters.secondary,"secondary"); 
                rtc.filters.exterior = convertTomlArrayToEigenMatrix(*ctrl_tbl["exterior"].as_array(),rtc.filters.exterior,"exterior"); 
                rtc.filters.inner_pupil_filt = convertTomlArrayToEigenMatrix(*ctrl_tbl["inner_pupil_filt"].as_array(),rtc.filters.inner_pupil_filt,"inner_pupil_filt");

            }catch (const std::exception& ex) {
                std::cerr << "Error processing rtc.pixels: " << ex.what() << std::endl;
                std::exit(1);
            }

            // Matricies
            try{
                rtc.matrices.szm = ctrl_tbl["szm"].value_or(0.0);
                rtc.matrices.sza = ctrl_tbl["sza"].value_or(0.0);
                rtc.matrices.szp = ctrl_tbl["szp"].value_or(0.0);
            
                rtc.matrices.I2A = convertTomlArrayToEigenMatrix(*ctrl_tbl["I2A"].as_array(),rtc.matrices.I2A,"I2A");
                //rtc.matrices.I2M = convertTomlArrayToEigenMatrix(*ctrl_tbl["I2M"].as_array(),rtc.matrices.I2M,"I2M");
                rtc.matrices.I2M_LO = convertTomlArrayToEigenMatrix(*ctrl_tbl["I2M_LO"].as_array(),rtc.matrices.I2M_LO,"I2M_LO");
                rtc.matrices.I2M_HO = convertTomlArrayToEigenMatrix(*ctrl_tbl["I2M_HO"].as_array(),rtc.matrices.I2M_HO,"I2M_HO");
                rtc.matrices.M2C = convertTomlArrayToEigenMatrix(*ctrl_tbl["M2C"].as_array(),rtc.matrices.M2C,"M2C");
                rtc.matrices.M2C_LO = convertTomlArrayToEigenMatrix(*ctrl_tbl["M2C_LO"].as_array(),rtc.matrices.M2C_LO,"M2C_LO");
                rtc.matrices.M2C_HO = convertTomlArrayToEigenMatrix(*ctrl_tbl["M2C_HO"].as_array(),rtc.matrices.M2C_HO,"M2C_HO");
                rtc.matrices.I2rms_sec = convertTomlArrayToEigenMatrix(*ctrl_tbl["I2rms_sec"].as_array(),rtc.matrices.I2rms_sec,"I2rms_sec");
                rtc.matrices.I2rms_ext = convertTomlArrayToEigenMatrix(*ctrl_tbl["I2rms_ext"].as_array(),rtc.matrices.I2rms_ext,"I2rms_ext");
            }catch(const std::exception& ex) {
                std::cerr << "Error processing rtc.matricies: " << ex.what() << std::endl;
                std::exit(1);
            }
            // reference intensities
            try{
                rtc.reference_pupils.I0 = convertTomlArrayToEigenMatrix(*ctrl_tbl["I0"].as_array(),rtc.reference_pupils.I0,"I0");
                rtc.reference_pupils.N0 = convertTomlArrayToEigenMatrix(*ctrl_tbl["N0"].as_array(),rtc.reference_pupils.N0,"NO");
                rtc.reference_pupils.norm_pupil = convertTomlArrayToEigenMatrix(*ctrl_tbl["norm_pupil"].as_array(),rtc.reference_pupils.norm_pupil,"norm_pupil");
                rtc.reference_pupils.intrn_flx_I0 = ctrl_tbl["intrn_flx_I0"].value_or(1.0);
                std::cout << "rtc.reference_pupils.intrn_flx_I0 " << rtc.reference_pupils.intrn_flx_I0 << std::endl;
            }catch(const std::exception& ex) {
                std::cerr << "Error processing rtc.reference_pupils " << ex.what() << std::endl;
                std::exit(1);
            }

            try{
                // method in reference_pupils to project them onto registered DM pixels and fill rtc.reference_pupils.norm_pupil_dm, rtc.reference_pupils.I0_dm
                rtc.reference_pupils.project_to_dm( rtc.matrices.I2A );  // does both I0 and N0
                // same for darks and bias (put on DM)
                rtc.reduction.project_to_dm( rtc.matrices.I2A );
                // project all the filters to get a weighted filter in dm space
                rtc.filters.project_to_dm( rtc.matrices.I2A );
            }catch(const std::exception& ex) {
                std::cerr << "Error processing projection of matricies to dm space " << ex.what() << std::endl;
                std::exit(1);
            }

            try{
                // limits float 
                rtc.limits.close_on_strehl_limit = ctrl_tbl["close_on_strehl_limit"].value_or(0.0);
                rtc.limits.open_on_strehl_limit = ctrl_tbl["open_on_strehl_limit"].value_or(0.0);
                rtc.limits.open_on_flux_limit = ctrl_tbl["open_on_flux_limit"].value_or(0.0);
                rtc.limits.open_on_dm_limit = ctrl_tbl["open_on_dm_limit"].value_or(0.0);
                rtc.limits.LO_offload_limit = ctrl_tbl["LO_offload_limit"].value_or(0.0);
            }catch(const std::exception& ex) {
                std::cerr << "Error with limits readin " << ex.what() << std::endl;
                std::exit(1);
            }
            // cam config when building interaction matrix 
            try{
                if (auto cam_node = ctrl_tbl["camera_config"]; cam_node && cam_node.is_table()) {
                    auto cam_tbl = *cam_node.as_table();

                    rtc.cam.fps = cam_tbl["fps"].value_or(std::string(""));
                    rtc.cam.gain = cam_tbl["gain"].value_or(std::string(""));
                    rtc.cam.testpattern = cam_tbl["testpattern"].value_or(std::string(""));

                    rtc.cam.fps  = cam_tbl["fps"].value_or(std::string(""));
                    rtc.cam.gain  = cam_tbl["gain"].value_or(std::string(""));
                    rtc.cam.testpattern = cam_tbl["testpattern"].value_or(std::string(""));
                    rtc.cam.bias = cam_tbl["bias"].value_or(std::string(""));
                    rtc.cam.flat = cam_tbl["flat"].value_or(std::string(""));
                    rtc.cam.imagetags = cam_tbl["imagetags"].value_or(std::string(""));
                    rtc.cam.led = cam_tbl["led"].value_or(std::string(""));
                    rtc.cam.events = cam_tbl["events"].value_or(std::string(""));
                    rtc.cam.extsynchro = cam_tbl["extsynchro"].value_or(std::string(""));
                    rtc.cam.rawimages = cam_tbl["rawimages"].value_or(std::string(""));
                    rtc.cam.cooling = cam_tbl["cooling"].value_or(std::string(""));
                    rtc.cam.mode = cam_tbl["mode"].value_or(std::string(""));
                    rtc.cam.resetwidth = cam_tbl["resetwidth"].value_or(std::string(""));
                    rtc.cam.nbreadworeset = cam_tbl["nbreadworeset"].value_or(std::string(""));
                    rtc.cam.cropping = cam_tbl["cropping"].value_or(std::string(""));
                    rtc.cam.cropping_columns = cam_tbl["cropping_columns"].value_or(std::string(""));
                    rtc.cam.cropping_rows = cam_tbl["cropping_rows"].value_or(std::string(""));
                    rtc.cam.aduoffset = cam_tbl["aduoffset"].value_or(std::string(""));
                }
            }
            catch(const std::exception& ex) {
                std::cerr << "Error with camera configuration readin" << ex.what() << std::endl;
                std::exit(1);
            }
        
    }
    }

    std::cout << "here here" << std::endl;

    try{    
        // init telemetry 
        rtc.telem = bdr_telem();

        // init contoller 
        //// heree
        //rtc.controller = bdr_controller(); 
        //int LO_nRows = rtc.matrices.I2M_LO.cols();//rtc_config.matrices.I2M_LO.rows();
        //int HO_nRows = rtc.matrices.I2M_HO.cols();//rtc_config.matrices.I2M_HO.rows();
        //std::cout << "LO_nRows = " << LO_nRows << ", HO_nRows = " << HO_nRows << std::endl;
        rtc.ctrl_LO_config = bdr_controller( 2 ); //LO_nRows ); 
        rtc.ctrl_HO_config = bdr_controller( 140 );//HO_nRows ); 

        std::cout << "in read in rtc.ctrl_LO_config.kp.size() = " << rtc.ctrl_LO_config.kp.size() << std::endl;

                
        // write method , back to toml, or json?


    }catch (const std::exception& ex) {
        std::cerr << "Error initializing telemetry or controller: " << ex.what() << std::endl;
        std::exit(1);
    }

    try{
        // Validate the master configuration.
        rtc.validate();
    }catch(const std::exception& ex) {
        std::cerr << "Error validating RTC configuration: " << ex.what() << std::endl;
        std::exit(1);
    }

    return rtc;
}


// stuff for on sky interaction data aquisition (19/8/25)
// Returns the current DM command vector length from SHM metadata.

// Returns the current DM command vector length from SHM metadata.
static inline int dm_cmd_len_unsafe() {
    return (dm_rtc.md ? static_cast<int>(dm_rtc.md->nelement) : 0);
}

// Initialize g_ol_offsets to zero vectors of length N (idempotent).
static bool ol_init_if_needed(int N_hint = -1) {
    auto cur = std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
    if (cur) return true;

    int N = (N_hint > 0) ? N_hint : dm_cmd_len_unsafe();
    if (N <= 0) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "[OL] init: DM command length unknown (N<=0). "
                         "Offsets remain null until DM SHM is attached.\n";
            warned = true;
        }
        return false;
    }

    // Build mutable, then publish as const
    auto z_mut = std::make_shared<OLOffsets>();
    z_mut->lo = Eigen::VectorXd::Zero(N);
    z_mut->ho = Eigen::VectorXd::Zero(N);
    std::shared_ptr<const OLOffsets> z_const = z_mut;
    std::atomic_store_explicit(&g_ol_offsets, z_const, std::memory_order_release);
    std::cerr << "[OL] init: initialized open-loop offsets with length " << N << ".\n";
    return true;
}

// Publish a new snapshot (single atomic store)
static inline void ol_publish(std::shared_ptr<const OLOffsets> nv_const) {
    std::atomic_store_explicit(&g_ol_offsets, std::move(nv_const), std::memory_order_release);
}

// Call once after DM SHM attach when DM command length is known.
void init_openloop_offsets(int dm_cmd_len) {
    (void)ol_init_if_needed(dm_cmd_len);
}

// Replace ONLY the LO offset (size must match DM length)
void set_openloop_offset_LO(const Eigen::Ref<const Eigen::VectorXd>& lo) {
    if (!ol_init_if_needed()) { std::cerr << "[OL] set_lo: not initialized.\n"; return; }
    auto cur = std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
    const int N = cur->lo.size();
    if (lo.size() != N) {
        std::cerr << "[OL] set_lo: size mismatch (got " << lo.size()
                  << ", expected " << N << ").\n";
        return;
    }
    auto nv_mut = std::make_shared<OLOffsets>(*cur);  // copy HO, replace LO
    nv_mut->lo = lo;
    std::shared_ptr<const OLOffsets> nv_const = nv_mut;
    ol_publish(std::move(nv_const));
}

// Replace ONLY the HO offset (size must match DM length)
void set_openloop_offset_HO(const Eigen::Ref<const Eigen::VectorXd>& ho) {
    if (!ol_init_if_needed()) { std::cerr << "[OL] set_ho: not initialized.\n"; return; }
    auto cur = std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
    const int N = cur->ho.size();
    if (ho.size() != N) {
        std::cerr << "[OL] set_ho: size mismatch (got " << ho.size()
                  << ", expected " << N << ").\n";
        return;
    }
    auto nv_mut = std::make_shared<OLOffsets>(*cur);  // copy LO, replace HO
    nv_mut->ho = ho;
    std::shared_ptr<const OLOffsets> nv_const = nv_mut;
    ol_publish(std::move(nv_const));
}

// Zero both offsets and publish atomically
void clear_openloop_offsets() {
    if (!ol_init_if_needed()) return;
    auto cur = std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
    auto nv_mut = std::make_shared<OLOffsets>(*cur);
    nv_mut->lo.setZero();
    nv_mut->ho.setZero();
    std::shared_ptr<const OLOffsets> nv_const = nv_mut;
    ol_publish(std::move(nv_const));
}

// Set one actuator in LO or HO open-loop offset vector.
// branch: "LO" or "HO" (case-insensitive)
// idx: 0-based actuator index
// Commander entrypoint: set one actuator open-loop offset (LO/HO) in-place.
// Accepts: ["HO", 65, 0.1]  
json ol_set_actuator(json args) {
    try {
        std::string branch;
        long long   idx_ll = -1;
        double      value  = 0.0;

        // Parse input ---------------------------------------------------------
        if (args.is_array()) {
            if (args.size() != 3)
                return json{{"ok", false}, {"error", "usage: [\"HO|LO\", idx, value]"}};

            branch = args.at(0).get<std::string>();

            // idx can be number or string; accept both
            if (args.at(1).is_number_integer() || args.at(1).is_number_unsigned())
                idx_ll = args.at(1).get<long long>();
            else if (args.at(1).is_string())
                idx_ll = std::stoll(args.at(1).get<std::string>());
            else
                return json{{"ok", false}, {"error", "idx must be integer"}};

            // value can be number or string; accept both
            if (args.at(2).is_number_float() || args.at(2).is_number_integer())
                value = args.at(2).get<double>();
            else if (args.at(2).is_string())
                value = std::stod(args.at(2).get<std::string>());
            else
                return json{{"ok", false}, {"error", "value must be numeric"}};
        }
        else if (args.is_object()) {
            if (!args.contains("branch") || !args.contains("idx") || !args.contains("value"))
                return json{{"ok", false}, {"error", "usage: {\"branch\":\"HO|LO\",\"idx\":N,\"value\":X}"}};

            branch = args["branch"].get<std::string>();

            if (args["idx"].is_number_integer() || args["idx"].is_number_unsigned())
                idx_ll = args["idx"].get<long long>();
            else if (args["idx"].is_string())
                idx_ll = std::stoll(args["idx"].get<std::string>());
            else
                return json{{"ok", false}, {"error", "idx must be integer"}};

            if (args["value"].is_number_float() || args["value"].is_number_integer())
                value = args["value"].get<double>();
            else if (args["value"].is_string())
                value = std::stod(args["value"].get<std::string>());
            else
                return json{{"ok", false}, {"error", "value must be numeric"}};
        }
        else if (args.is_string()) {
            std::istringstream iss(args.get<std::string>());
            if (!(iss >> branch >> idx_ll >> value))
                return json{{"ok", false}, {"error", "usage: \"HO 65 0.1\""}};
        }
        else {
            return json{{"ok", false}, {"error", "usage: [\"HO|LO\", idx, value] or {\"branch\":\"HO|LO\",\"idx\":N,\"value\":X}"}};
        }

        // Validate ------------------------------------------------------------
        if (idx_ll < 0)
            return json{{"ok", false}, {"error", "idx must be >= 0"}};

        // Ensure offsets are initialized (requires DM SHM size known)
        if (!ol_init_if_needed())
            return json{{"ok", false}, {"error", "offsets not initialized (DM length unknown)"}};

        auto cur = std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
        if (!cur)
            return json{{"ok", false}, {"error", "internal null snapshot"}};

        const int N   = static_cast<int>(cur->lo.size());   // == cur->ho.size()
        const int idx = static_cast<int>(idx_ll);
        if (idx >= N)
            return json{{"ok", false}, {"error", "idx out of range", {"N", N}}};

        // Apply ----------------------------------------------------------------
        auto nv_mut = std::make_shared<OLOffsets>(*cur);
        const bool isHO = (!branch.empty() && (branch[0] == 'H' || branch[0] == 'h'));

        const double old_val = isHO ? nv_mut->ho(idx) : nv_mut->lo(idx);
        if (isHO) nv_mut->ho(idx) = value;
        else      nv_mut->lo(idx) = value;

        std::shared_ptr<const OLOffsets> nv_const = nv_mut;
        std::atomic_store_explicit(&g_ol_offsets, nv_const, std::memory_order_release);

        return json{{"ok", true},
                    {"branch", isHO ? "HO" : "LO"},
                    {"idx", idx},
                    {"old", old_val},
                    {"value", value},
                    {"N", N}};
    }
    catch (const std::exception& e) {
        return json{{"ok", false}, {"error", std::string("parse/apply: ") + e.what()}};
    }
}

/////////////////

//uncomment and build July 2025 AIV
void change_boxcar(int cmd) {
    try {
        int new_val = cmd; //std::stoi(cmd);
        if (new_val > 0 && new_val < 1000) {
            global_boxcar.store(new_val); //atomic threadsafe
            std::cout << "Boxcar window updated to " << new_val << std::endl;
        } else {
            std::cerr << "Boxcar value out of allowed range (1–999)" << std::endl;
        }
    } catch (...) {
        std::cerr << "Invalid boxcar input: " << cmd << std::endl;
    }
}

// test in TTR115.0035 (not tested yet)
// JSON-returning version compatible with Commander
nlohmann::json poll_telem_vector(std::string name) {
    static const std::unordered_map<std::string,
        boost::circular_buffer<Eigen::VectorXd> bdr_telem::*> telemetry_map = {
        {"img",    &bdr_telem::img},
        {"img_dm", &bdr_telem::img_dm},
        {"signal", &bdr_telem::signal},
        {"e_LO",   &bdr_telem::e_LO},
        {"u_LO",   &bdr_telem::u_LO},
        {"e_HO",   &bdr_telem::e_HO},
        {"u_HO",   &bdr_telem::u_HO},
        {"c_LO",   &bdr_telem::c_LO},
        {"c_HO",   &bdr_telem::c_HO},
        {"c_inj",   &bdr_telem::c_inj},
    };

    nlohmann::json j;

    auto it = telemetry_map.find(name);
    if (it == telemetry_map.end()) {
        // consistent, explicit error payload
        return nlohmann::json{{"ok", false}, {"error", "unknown field"}, {"field", name}};
    }

    std::lock_guard<std::mutex> lk(telemetry_mutex);

    const auto& buffer = rtc_config.telem.*(it->second);
    const auto n = buffer.size();
    if (n < 2) {
        // “no data yet” -> null data with ok=false
        return nlohmann::json{{"ok", false}, {"error", "insufficient data"}, {"size", n}};
    }

    const Eigen::VectorXd& v = buffer[n - 2];
    // Convert Eigen -> std::vector<double> -> JSON array
    std::vector<double> out(v.data(), v.data() + v.size());

    j = nlohmann::json{
        {"ok",   true},
        {"name", name},
        {"size", v.size()},
        {"data", out}
    };
    return j;
}


// // test in TTR115.0035 (not tested yet)
// JSON-returning scalar poll (Commander-friendly)
nlohmann::json poll_telem_scalar(std::string name) {
    // Only true scalars here. (LO/HO states are ints—expose via a separate command if needed.)
    static const std::unordered_map<std::string,
        boost::circular_buffer<double> bdr_telem::*> scalar_map = {
        {"rmse_est", &bdr_telem::rmse_est},
        {"snr",      &bdr_telem::snr},
        // add more scalar fields here if you have them
    };

    auto it = scalar_map.find(name);
    if (it == scalar_map.end()) {
        return nlohmann::json{{"ok", false}, {"error", "unknown field"}, {"field", name}};
    }

    std::lock_guard<std::mutex> lk(telemetry_mutex);

    const auto& buffer = rtc_config.telem.*(it->second);
    const auto n = buffer.size();
    if (n < 2) {
        return nlohmann::json{{"ok", false}, {"error", "insufficient data"}, {"size", n}};
    }

    const double val = buffer[n - 2];
    return nlohmann::json{
        {"ok",   true},
        {"name", name},
        {"data", val}
    };
}





// // these are outdated since the darks now go through Cred1 server (camera)
// void new_dark_and_bias() {
//     try {
//         std::cout << "[capture_dark_and_bias] Starting new dark/bias capture..." << std::endl;

//         // Pause the RTC loop
//         pauseRTC();
//         std::cout << "[capture_dark_and_bias] RTC paused." << std::endl;

//         // Turn off the source
//         send_mds_cmd("off SBB");
//         std::cout << "[capture_dark_and_bias] Light source turned off." << std::endl;
//         std::this_thread::sleep_for(std::chrono::seconds(1));

//         // Record original FPS and gain
//         float original_fps = get_float_cam_param("fps raw");
//         float original_gain = get_float_cam_param("gain raw");

//         // Set FPS to maximum (1730 Hz) for bias measurement
//         send_cam_cmd("set fps 1730");
//         std::this_thread::sleep_for(std::chrono::seconds(1));

//         const size_t nframes = 1000;
//         size_t totalPixels = static_cast<size_t>(subarray.md->size[0]) * static_cast<size_t>(subarray.md->size[1]);

//         Eigen::VectorXd bias_sum = Eigen::VectorXd::Zero(totalPixels);

//         // --- Capture bias frames ---
//         for (size_t i = 0; i < nframes; ++i) {
//             catch_up_with_sem(&subarray, 1);
//             ImageStreamIO_semwait(&subarray, 1);

//             //uint16_t* raw = subarray.array.UI16;
//             int32_t* raw = subarray.array.SI32;
//             //Eigen::Map<const Eigen::Array<uint16_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
//             Eigen::Map<const Eigen::Array<int32_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
//             bias_sum += rawArr.cast<double>().matrix();
//         }

//         Eigen::VectorXd bias = bias_sum / static_cast<double>(nframes);

//         std::cout << "[capture_dark_and_bias] Bias frames captured." << std::endl;

//         // Restore original FPS
//         send_cam_cmd("set fps " + std::to_string(static_cast<int>(original_fps)));
//         std::this_thread::sleep_for(std::chrono::seconds(1));

//         // --- Capture dark frames ---
//         Eigen::VectorXd dark_sum = Eigen::VectorXd::Zero(totalPixels);

//         for (size_t i = 0; i < nframes; ++i) {
//             catch_up_with_sem(&subarray, 1);
//             ImageStreamIO_semwait(&subarray, 1);

//             //uint16_t* raw = subarray.array.UI16;
//             //Eigen::Map<const Eigen::Array<uint16_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
//             int32_t* raw = subarray.array.SI32;
//             Eigen::Map<const Eigen::Array<int32_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
//             dark_sum += rawArr.cast<double>().matrix();
//         }

//         Eigen::VectorXd dark = dark_sum / static_cast<double>(nframes);

//         std::cout << "[capture_dark_and_bias] Dark frames captured." << std::endl;

//         // Turn the source back on
//         send_mds_cmd("on SBB");
//         std::this_thread::sleep_for(std::chrono::seconds(2));
//         std::cout << "[capture_dark_and_bias] Light source turned back on." << std::endl;

//         // Compute raw dark: (dark - bias) (in ADU)
//         Eigen::VectorXd raw_dark = dark - bias;

//         // Save to rtc_config
//         rtc_config.reduction.bias = bias;
//         rtc_config.reduction.dark = raw_dark * (original_fps / original_gain); // Units: ADU/s/gain

//         rtc_config.reduction.bias_dm = rtc_config.matrices.I2A * rtc_config.reduction.bias;
//         rtc_config.reduction.dark_dm = rtc_config.matrices.I2A * rtc_config.reduction.dark;

//         rtc_config.dark_dm_runtime = rtc_config.matrices.I2A * raw_dark;

//         std::cout << "[capture_dark_and_bias] Updated reduction parameters successfully." << std::endl;

//         // Resume the RTC
//         resumeRTC();
//         std::cout << "[capture_dark_and_bias] RTC resumed." << std::endl;

//     } catch (const std::exception& ex) {
//         std::cerr << "[capture_dark_and_bias] Exception occurred: " << ex.what() << std::endl;
//         resumeRTC(); // Ensure we resume RTC even on failure
//     }
// }



// void capture_dark_and_bias() {
//     try {
//         pauseRTC();  // Pause RTC while capturing frames
//         std::cout << "[capture_dark_and_bias] RTC paused for dark and bias acquisition." << std::endl;

//         new_dark_and_bias();

//         resumeRTC(); // Resume RTC afterwards
//         std::cout << "[capture_dark_and_bias] RTC resumed after dark and bias acquisition." << std::endl;
//     }
//     catch (const std::exception& ex) {
//         std::cerr << "[capture_dark_and_bias] Exception caught: " << ex.what() << std::endl;
//         resumeRTC(); // Try to resume even on error
//     }
// }





///////////////////////////////////////////
///// untested version 2 of getter and setters! 
struct FieldHandle {
    std::string type;  // for introspection ("double","int","string","vector","matrix",…)
    std::function<nlohmann::json()> get;
    std::function<bool(const nlohmann::json&)> set; // nullptr => read-only
};

// static inline bool starts_with(std::string_view s, std::string_view p) {
//     return s.size() >= p.size() && 0 == s.compare(0, p.size(), p);
// }
// static inline std::string after_prefix(std::string_view s, std::string_view p) {
//     return std::string(s.substr(p.size()));
// }

// struct CtrlRef {
//     std::unique_ptr<Controller>* ptr;  // rtc_config.ctrl_LO / ctrl_HO
//     std::string name;                  // parameter name after the prefix
// };

// static std::optional<CtrlRef> parse_ctrl_field(std::string_view field) {
//     if (starts_with(field, "ctrl_LO."))
//         return CtrlRef{ &rtc_config.ctrl_LO, after_prefix(field, "ctrl_LO.") };
//     if (starts_with(field, "ctrl_HO."))
//         return CtrlRef{ &rtc_config.ctrl_HO, after_prefix(field, "ctrl_HO.") };
//     return std::nullopt;
// }

// static json ctrl_get_field(const CtrlRef& ref) {
//     std::lock_guard<std::mutex> lk(ctrl_mutex);
//     if (!*ref.ptr) return json{{"error","controller not initialized"}};

//     if (ref.name == "ctrl_type") {
//         return (*ref.ptr)->get_type();
//     }

//     Controller::Param p;
//     try {
//         p = (*ref.ptr)->get_parameter(ref.name);
//     } catch (const std::exception& e) {
//         json names = json::array();
//         for (const auto& kv : (*ref.ptr)->list_parameters()) names.push_back(kv.first);
//         return json{{"error", std::string("unknown parameter: ")+ref.name}, {"available", names}};
//     }

//     if (std::holds_alternative<Eigen::VectorXd>(p)) {
//         return eigen_vector_to_json(std::get<Eigen::VectorXd>(p));
//     }
//     if (std::holds_alternative<Eigen::MatrixXd>(p)) {
//         return eigen_matrix_to_json(std::get<Eigen::MatrixXd>(p));
//     }
//     return json{{"error","unsupported parameter kind"}};
// }

// static json ctrl_set_field(const CtrlRef& ref, const json& value) {
//     std::lock_guard<std::mutex> lk(ctrl_mutex);
//     if (!*ref.ptr) return json{{"error","controller not initialized"}};
//     if (ref.name == "ctrl_type") return json{{"error","ctrl_type is read-only"}};

//     // Try vector
//     {
//         Eigen::VectorXd v;
//         if (json_to_eigen_vector(value, v)) {
//             try { (*ref.ptr)->set_parameter(ref.name, v); return json{{"status","ok"}}; }
//             catch (const std::exception& e) { return json{{"error", e.what()}}; }
//         }
//     }
//     // Try matrix
//     {
//         Eigen::MatrixXd M;
//         if (json_to_eigen_matrix(value, M)) {
//             try { (*ref.ptr)->set_parameter(ref.name, M); return json{{"status","ok"}}; }
//             catch (const std::exception& e) { return json{{"error", e.what()}}; }
//         }
//     }
//     return json{{"error","value is neither vector nor matrix"}};
// }



// ---- Eigen <-> JSON helpers ----
template<typename EigenVec>
static nlohmann::json eigen_vector_to_json(const EigenVec& v) {
    using S = typename EigenVec::Scalar;
    if constexpr (std::is_integral_v<S>) {
        std::vector<long long> out(v.size());
        for (Eigen::Index i = 0; i < v.size(); ++i) out[i] = static_cast<long long>(v(i));
        return out;
    } else {
        std::vector<double> out(v.size());
        for (Eigen::Index i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v(i));
        return out;
    }
}

template<typename EigenMat>
static nlohmann::json eigen_matrix_to_json(const EigenMat& M) {
    using S = typename EigenMat::Scalar;
    nlohmann::json rows = nlohmann::json::array();
    for (Eigen::Index r = 0; r < M.rows(); ++r) {
        if constexpr (std::is_integral_v<S>) {
            std::vector<long long> row(M.cols());
            for (Eigen::Index c = 0; c < M.cols(); ++c) row[c] = static_cast<long long>(M(r,c));
            rows.push_back(row);
        } else {
            std::vector<double> row(M.cols());
            for (Eigen::Index c = 0; c < M.cols(); ++c) row[c] = static_cast<double>(M(r,c));
            rows.push_back(row);
        }
    }
    return rows;
}

template<typename EigenVec>
static bool json_to_eigen_vector(const nlohmann::json& j, EigenVec& out) {
    if (!j.is_array()) return false;
    using S = typename EigenVec::Scalar;
    const auto n = j.size();
    out.resize(static_cast<Eigen::Index>(n));
    try {
        for (size_t i = 0; i < n; ++i) out(static_cast<Eigen::Index>(i)) = static_cast<S>(j.at(i).get<double>());
        return true;
    } catch (...) { return false; }
}

template<typename EigenMat>
static bool json_to_eigen_matrix(const nlohmann::json& j, EigenMat& out) {
    if (!j.is_array() || j.empty() || !j.front().is_array()) return false;
    const auto rows = j.size();
    const auto cols = j.front().size();
    for (const auto& row : j) if (!row.is_array() || row.size() != cols) return false;
    out.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
    try {
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
                    static_cast<typename EigenMat::Scalar>(j[r][c].get<double>());
        return true;
    } catch (...) { return false; }
}


static Controller* pick_ctrl(const bdr_rtc_config* cfg, const char* which) {
    if (!which) return nullptr;
    const bool lo = (which[0]=='L'||which[0]=='l') && (which[1]=='O'||which[1]=='o');
    const bool ho = (which[0]=='H'||which[0]=='h') && (which[1]=='O'||which[1]=='o');
    if (lo) return cfg->ctrl_LO ? cfg->ctrl_LO.get() : nullptr;
    if (ho) return cfg->ctrl_HO ? cfg->ctrl_HO.get() : nullptr;
    return nullptr;
}


// ---- Getter-only factories ----
template<typename T>
static FieldHandle make_scalar_field_getter(T bdr_rtc_config::* ptr, const char* type_name) {
    return FieldHandle{
        type_name,
        [ptr]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return rtc_config.*ptr; },
        nullptr
    };
}
template<typename Sub, typename T>
static FieldHandle make_nested_scalar_getter(Sub bdr_rtc_config::* sub, T Sub::* mem, const char* type_name) {
    return FieldHandle{
        type_name,
        [sub, mem]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return (rtc_config.*sub).*mem; },
        nullptr
    };
}
template<typename T>
static FieldHandle make_vector_field_getter(std::vector<T> bdr_rtc_config::* ptr, const char* type_name) {
    return FieldHandle{
        type_name,
        [ptr]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return rtc_config.*ptr; },
        nullptr
    };
}
template<typename Sub, typename EigenVec>
static FieldHandle make_nested_eigen_vector_getter(Sub bdr_rtc_config::* sub, EigenVec Sub::* mem, const char* tn="vector") {
    return FieldHandle{
        tn,
        [sub, mem]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return eigen_vector_to_json((rtc_config.*sub).*mem); },
        nullptr
    };
}
template<typename Sub, typename EigenMat>
static FieldHandle make_nested_eigen_matrix_getter(Sub bdr_rtc_config::* sub, EigenMat Sub::* mem, const char* tn="matrix") {
    return FieldHandle{
        tn,
        [sub, mem]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return eigen_matrix_to_json((rtc_config.*sub).*mem); },
        nullptr
    };
}
template<typename EigenVec>
static FieldHandle make_eigen_vector_field_getter(EigenVec bdr_rtc_config::* mem, const char* tn="vector") {
    return FieldHandle{
        tn,
        [mem]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return eigen_vector_to_json(rtc_config.*mem); },
        nullptr
    };
}
template<typename EigenMat>
static FieldHandle make_eigen_matrix_field_getter(EigenMat bdr_rtc_config::* mem, const char* tn="matrix") {
    return FieldHandle{
        tn,
        [mem]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return eigen_matrix_to_json(rtc_config.*mem); },
        nullptr
    };
}

// ---- Read/Write factories (use these for fields you want to mutate) ----
template<typename T>
static FieldHandle make_scalar_field_rw(T bdr_rtc_config::* ptr, const char* type_name) {
    return FieldHandle{
        type_name,
        [ptr]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return rtc_config.*ptr; },
        [ptr](const nlohmann::json& j) -> bool {
            try { T v = j.get<T>(); std::lock_guard<std::mutex> lk(rtc_mutex); rtc_config.*ptr = v; return true; }
            catch (...) { return false; }
        }
    };
}
template<typename Sub, typename T>
static FieldHandle make_nested_scalar_rw(Sub bdr_rtc_config::* sub, T Sub::* mem, const char* type_name) {
    return FieldHandle{
        type_name,
        [sub, mem]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return (rtc_config.*sub).*mem; },
        [sub, mem](const nlohmann::json& j) -> bool {
            try { T v = j.get<T>(); std::lock_guard<std::mutex> lk(rtc_mutex); (rtc_config.*sub).*mem = v; return true; }
            catch (...) { return false; }
        }
    };
}
template<typename T>
static FieldHandle make_vector_rw(std::vector<T> bdr_rtc_config::* ptr, const char* type_name) {
    return FieldHandle{
        type_name,
        [ptr]() -> nlohmann::json { std::lock_guard<std::mutex> lk(rtc_mutex); return rtc_config.*ptr; },
        [ptr](const nlohmann::json& j) -> bool {
            if (!j.is_array()) return false;
            try {
                std::vector<T> v = j.get<std::vector<T>>();
                std::lock_guard<std::mutex> lk(rtc_mutex);
                rtc_config.*ptr = std::move(v);
                return true;
            } catch (...) { return false; }
        }
    };
}


// ---- Read/Write factories for Eigen types ----
template<typename EigenVec>
static FieldHandle make_eigen_vector_field_rw(EigenVec bdr_rtc_config::* mem,
                                              const char* tn = "vector<double>") {
    return FieldHandle{
        tn,
        // Getter -> JSON
        [mem]() -> nlohmann::json {
            std::lock_guard<std::mutex> lk(rtc_mutex);
            return eigen_vector_to_json(rtc_config.*mem);
        },
        // Setter <- JSON
        [mem](const nlohmann::json& j) -> bool {
            try {
                EigenVec tmp;
                if (!json_to_eigen_vector(j, tmp)) return false;
                std::lock_guard<std::mutex> lk(rtc_mutex);
                (rtc_config.*mem) = std::move(tmp);
                return true;
            } catch (...) {
                return false;
            }
        }
    };
}

template<typename EigenMat>
static FieldHandle make_eigen_matrix_field_rw(EigenMat bdr_rtc_config::* mem,
                                              const char* tn = "matrix<double>") {
    return FieldHandle{
        tn,
        // Getter -> JSON
        [mem]() -> nlohmann::json {
            std::lock_guard<std::mutex> lk(rtc_mutex);
            return eigen_matrix_to_json(rtc_config.*mem);
        },
        // Setter <- JSON
        [mem](const nlohmann::json& j) -> bool {
            try {
                EigenMat tmp;
                if (!json_to_eigen_matrix(j, tmp)) return false;
                std::lock_guard<std::mutex> lk(rtc_mutex);
                (rtc_config.*mem) = std::move(tmp);
                return true;
            } catch (...) {
                return false;
            }
        }
    };
}


template<typename Sub, typename EigenVec>
static FieldHandle make_nested_eigen_vector_rw(Sub bdr_rtc_config::* sub, EigenVec Sub::* mem, const char* tn="vector<double>") {
    return FieldHandle{
        tn,
        [sub, mem]() -> nlohmann::json {
            std::lock_guard<std::mutex> lk(rtc_mutex);
            return eigen_vector_to_json((rtc_config.*sub).*mem);
        },
        [sub, mem](const nlohmann::json& j) -> bool {
            typename std::remove_reference_t<decltype((rtc_config.*sub).*mem)> tmp;
            if (!json_to_eigen_vector(j, tmp)) return false;
            std::lock_guard<std::mutex> lk(rtc_mutex);
            (rtc_config.*sub).*mem = std::move(tmp);
            return true;
        }
    };
}
template<typename Sub, typename EigenMat>
static FieldHandle make_nested_eigen_matrix_rw(Sub bdr_rtc_config::* sub, EigenMat Sub::* mem, const char* tn="matrix<double>") {
    return FieldHandle{
        tn,
        [sub, mem]() -> nlohmann::json {
            std::lock_guard<std::mutex> lk(rtc_mutex);
            return eigen_matrix_to_json((rtc_config.*sub).*mem);
        },
        [sub, mem](const nlohmann::json& j) -> bool {
            typename std::remove_reference_t<decltype((rtc_config.*sub).*mem)> tmp;
            if (!json_to_eigen_matrix(j, tmp)) return false;
            std::lock_guard<std::mutex> lk(rtc_mutex);
            (rtc_config.*sub).*mem = std::move(tmp);
            return true;
        }
    };
}

// --------- Registry of exposed fields on rtc_config ---------
// Telemetry ring buffers (rtc_config.telem.*) are intentionally omitted; you already access them via poll_telem_*.

static const std::unordered_map<std::string, FieldHandle> RTC_FIELDS = {
    // ===== state (RO) =====
    {"state.DM_flat",            make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::DM_flat, "string")},
    {"state.signal_space",       make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::signal_space, "string")},
    {"state.LO",                 make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::LO, "int")},
    {"state.controller_type",    make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::controller_type, "string")},
    {"state.inverse_method_LO",  make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::inverse_method_LO, "string")},
    {"state.inverse_method_HO",  make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::inverse_method_HO, "string")},
    {"state.phasemask",          make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::phasemask, "string")},
    {"state.auto_close",         make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::auto_close, "int")},
    {"state.auto_open",          make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::auto_open, "int")},
    {"state.auto_tune",          make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::auto_tune, "int")},
    {"state.take_telemetry",     make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::take_telemetry, "int")},
    {"state.simulation_mode",    make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::simulation_mode, "int")},

    // ===== reduction (RO) =====
    {"reduction.bias",           make_nested_eigen_vector_getter(&bdr_rtc_config::reduction, &bdr_reduction::bias)},
    {"reduction.bias_dm",        make_nested_eigen_vector_getter(&bdr_rtc_config::reduction, &bdr_reduction::bias_dm)},
    {"reduction.dark",           make_nested_eigen_vector_getter(&bdr_rtc_config::reduction, &bdr_reduction::dark)},
    {"reduction.dark_dm",        make_nested_eigen_vector_getter(&bdr_rtc_config::reduction, &bdr_reduction::dark_dm)},

    // ===== pixels (RO) =====
    {"pixels.crop_pixels",       make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::crop_pixels, "vector<int16>")},
    {"pixels.bad_pixels",        make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::bad_pixels, "vector<int32>")},
    {"pixels.pupil_pixels",      make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::pupil_pixels, "vector<int32>")},
    {"pixels.interior_pixels",   make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::interior_pixels, "vector<int32>")},
    {"pixels.secondary_pixels",  make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::secondary_pixels, "vector<int32>")},
    {"pixels.exterior_pixels",   make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::exterior_pixels, "vector<int32>")},

    // ===== reference_pupils (RW) =====
    {"reference_pupils.I0",           make_nested_eigen_vector_rw(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::I0)}, //make_nested_eigen_vector_getter(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::I0)},
    {"reference_pupils.N0",           make_nested_eigen_vector_rw(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::N0)},
    {"reference_pupils.norm_pupil",   make_nested_eigen_vector_rw(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::norm_pupil)},
    {"reference_pupils.norm_pupil_dm",make_nested_eigen_vector_rw(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::norm_pupil_dm)},
    {"reference_pupils.I0_dm",        make_nested_eigen_vector_rw(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::I0_dm)},

    // ===== matrices (RO) =====
    {"matrices.I2A",    make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::I2A)},
    {"matrices.I2M",    make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::I2M)},
    {"matrices.I2M_LO", make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::I2M_LO)},
    {"matrices.I2M_HO", make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::I2M_HO)},
    {"matrices.M2C",    make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::M2C)},
    {"matrices.M2C_LO", make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::M2C_LO)},
    {"matrices.M2C_HO", make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::M2C_HO)},
    {"matrices.I2rms_sec",  make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::I2rms_sec)},
    {"matrices.I2rms_ext",  make_nested_eigen_matrix_rw(&bdr_rtc_config::matrices, &bdr_matricies::I2rms_ext)},
    {"matrices.szm",        make_nested_scalar_rw(&bdr_rtc_config::matrices, &bdr_matricies::szm, "int")},
    {"matrices.sza",        make_nested_scalar_rw(&bdr_rtc_config::matrices, &bdr_matricies::sza, "int")},
    {"matrices.szp",        make_nested_scalar_rw(&bdr_rtc_config::matrices, &bdr_matricies::szp, "int")},
    // {"matrices.I2A",        make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2A)},
    // {"matrices.I2M",        make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2M)},
    // {"matrices.I2M_LO",     make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2M_LO)},
    // {"matrices.I2M_HO",     make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2M_HO)},
    // {"matrices.M2C",        make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::M2C)},
    // {"matrices.M2C_LO",     make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::M2C_LO)},
    // {"matrices.M2C_HO",     make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::M2C_HO)},
    // {"matrices.I2rms_sec",  make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2rms_sec)},
    // {"matrices.I2rms_ext",  make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2rms_ext)},
    // {"matrices.szm",        make_nested_scalar_getter(&bdr_rtc_config::matrices, &bdr_matricies::szm, "int")},
    // {"matrices.sza",        make_nested_scalar_getter(&bdr_rtc_config::matrices, &bdr_matricies::sza, "int")},
    // {"matrices.szp",        make_nested_scalar_getter(&bdr_rtc_config::matrices, &bdr_matricies::szp, "int")},

    // ===== controller configs (RW ) =====
    //// uncomment / fix after controller class upgrade verified 20-8-25
    // {"ctrl_LO_config.kp",           make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::kp)},
    // {"ctrl_LO_config.ki",           make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::ki)},
    // {"ctrl_LO_config.kd",           make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::kd)},
    // {"ctrl_LO_config.lower_limits", make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::lower_limits)},
    // {"ctrl_LO_config.upper_limits", make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::upper_limits)},
    // {"ctrl_LO_config.set_point",    make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::set_point)},

    // {"ctrl_HO_config.kp",           make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::kp)},
    // {"ctrl_HO_config.ki",           make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::ki)},
    // {"ctrl_HO_config.kd",           make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::kd)},
    // {"ctrl_HO_config.lower_limits", make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::lower_limits)},
    // {"ctrl_HO_config.upper_limits", make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::upper_limits)},
    // {"ctrl_HO_config.set_point",    make_nested_eigen_vector_rw(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::set_point)},

    // ===== limits (RW ) =====
    {"limits.close_on_strehl_limit", make_nested_scalar_rw(&bdr_rtc_config::limits, &bdr_limits::close_on_strehl_limit, "float")},
    {"limits.open_on_strehl_limit",  make_nested_scalar_rw(&bdr_rtc_config::limits, &bdr_limits::open_on_strehl_limit, "float")},
    {"limits.open_on_flux_limit",    make_nested_scalar_rw(&bdr_rtc_config::limits, &bdr_limits::open_on_flux_limit, "float")},
    {"limits.open_on_dm_limit",      make_nested_scalar_rw(&bdr_rtc_config::limits, &bdr_limits::open_on_dm_limit, "float")},
    {"limits.LO_offload_limit",      make_nested_scalar_rw(&bdr_rtc_config::limits, &bdr_limits::LO_offload_limit, "float")},

    // ===== cam (RO; strings control acquisition pipeline externally) =====
    {"cam.fps",               make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::fps, "string")},
    {"cam.gain",              make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::gain, "string")},
    {"cam.testpattern",       make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::testpattern, "string")},
    {"cam.bias",              make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::bias, "string")},
    {"cam.flat",              make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::flat, "string")},
    {"cam.imagetags",         make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::imagetags, "string")},
    {"cam.led",               make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::led, "string")},
    {"cam.events",            make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::events, "string")},
    {"cam.extsynchro",        make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::extsynchro, "string")},
    {"cam.rawimages",         make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::rawimages, "string")},
    {"cam.cooling",           make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::cooling, "string")},
    {"cam.mode",              make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::mode, "string")},
    {"cam.resetwidth",        make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::resetwidth, "string")},
    {"cam.nbreadworeset",     make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::nbreadworeset, "string")},
    {"cam.cropping",          make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::cropping, "string")},
    {"cam.cropping_columns",  make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::cropping_columns, "string")},
    {"cam.cropping_rows",     make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::cropping_rows, "string")},
    {"cam.aduoffset",         make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::aduoffset, "string")},

    // ===== filters (RO) =====
    {"filters.bad_pixel_mask",     make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::bad_pixel_mask, "vector<uint8>")},
    {"filters.bad_pixel_mask_dm",  make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::bad_pixel_mask_dm, "vector<float>")},
    {"filters.pupil",              make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::pupil, "vector<uint8>")},
    {"filters.pupil_dm",           make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::pupil_dm, "vector<float>")},
    {"filters.secondary",          make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::secondary, "vector<uint8>")},
    {"filters.secondary_dm",       make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::secondary_dm, "vector<float>")},
    {"filters.exterior",           make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::exterior, "vector<uint8>")},
    {"filters.exterior_dm",        make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::exterior_dm, "vector<float>")},
    {"filters.inner_pupil_filt",   make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::inner_pupil_filt, "vector<uint8>")},
    {"filters.inner_pupil_filt_dm",make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::inner_pupil_filt_dm, "vector<float>")},

    // ===== derived/runtime (top-level) =====
    {"zeroCmd",              make_eigen_vector_field_getter(&bdr_rtc_config::zeroCmd, "vector<double>")}, // RO by default
    {"fps",                  make_scalar_field_rw(&bdr_rtc_config::fps,   "double")},  // RW
    {"gain",                 make_scalar_field_rw(&bdr_rtc_config::gain,  "double")},  // RW
    {"scale",                make_scalar_field_rw(&bdr_rtc_config::scale, "double")},  // RW
    {"I2M_LO_runtime",       make_eigen_matrix_field_rw(&bdr_rtc_config::I2M_LO_runtime, "matrix<double>")},//make_eigen_matrix_field_getter(&bdr_rtc_config::I2M_LO_runtime, "matrix<double>")},
    {"I2M_HO_runtime",       make_eigen_matrix_field_getter(&bdr_rtc_config::I2M_HO_runtime, "matrix<double>")},//make_eigen_matrix_field_getter(&bdr_rtc_config::I2M_HO_runtime, "matrix<double>")},
    {"N0_dm_runtime",        make_eigen_vector_field_rw(&bdr_rtc_config::N0_dm_runtime, "vector<double>")},
    {"I0_dm_runtime",        make_eigen_vector_field_rw(&bdr_rtc_config::I0_dm_runtime, "vector<double>")},
    //{"dark_dm_runtime",      make_eigen_vector_field_getter(&bdr_rtc_config::dark_dm_runtime, "vector<double>")},
    {"sec_idx",              make_scalar_field_getter(&bdr_rtc_config::sec_idx, "int")},
    {"m_s_runtime",          make_scalar_field_getter(&bdr_rtc_config::m_s_runtime, "double")},
    {"b_s_runtime",          make_scalar_field_getter(&bdr_rtc_config::b_s_runtime, "double")},

    //// ===== runtime controllers (RO unless you explicitly want to poke internals) =====
    //// uncomment / fix after controller class upgrade verified 20-8-25
    // {"ctrl_LO.ctrl_type",    make_nested_scalar_getter(&bdr_rtc_config::ctrl_LO, &PIDController::ctrl_type, "string")},
    // {"ctrl_LO.kp",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::kp)},
    // {"ctrl_LO.ki",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::ki)},
    // {"ctrl_LO.kd",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::kd)},
    // {"ctrl_LO.lower_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::lower_limits)},
    // {"ctrl_LO.upper_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::upper_limits)},
    // {"ctrl_LO.set_point",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::set_point)},
    // {"ctrl_LO.output",       make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::output)},
    // {"ctrl_LO.integrals",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::integrals)},
    // {"ctrl_LO.prev_errors",  make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::prev_errors)},

    // {"ctrl_HO.ctrl_type",    make_nested_scalar_getter(&bdr_rtc_config::ctrl_HO, &PIDController::ctrl_type, "string")},
    // {"ctrl_HO.kp",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::kp)},
    // {"ctrl_HO.ki",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::ki)},
    // {"ctrl_HO.kd",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::kd)},
    // {"ctrl_HO.lower_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::lower_limits)},
    // {"ctrl_HO.upper_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::upper_limits)},
    // {"ctrl_HO.set_point",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::set_point)},
    // {"ctrl_HO.output",       make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::output)},
    // {"ctrl_HO.integrals",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::integrals)},
    // {"ctrl_HO.prev_errors",  make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::prev_errors)}

    // ===== inj_signal (RW) =====
    {"inj_signal.enabled",        make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::enabled, "int")},
    //{"inj_signal.space",          make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::space, "string")},   // , not used anymore
    {"inj_signal.basis",          make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::basis, "string")},
    {"inj_signal.basis_index",    make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::basis_index, "int")},
    {"inj_signal.amplitude",      make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::amplitude, "double")},
    {"inj_signal.waveform",       make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::waveform, "string")}, // "sine|square|step|chirp|prbs|none"
    {"inj_signal.freq_hz",        make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::freq_hz, "double")},
    {"inj_signal.phase_deg",      make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::phase_deg, "double")},
    {"inj_signal.duty",           make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::duty, "double")},
    {"inj_signal.t_start_s",      make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::t_start_s, "double")},
    {"inj_signal.t_stop_s",       make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::t_stop_s, "double")},
    {"inj_signal.latency_frames", make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::latency_frames, "int")},
    {"inj_signal.hold_frames",    make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::hold_frames, "int")},
    {"inj_signal.prbs_seed",      make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::prbs_seed, "uint32")},
    {"inj_signal.chirp_f0",       make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::chirp_f0, "double")},
    {"inj_signal.chirp_f1",       make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::chirp_f1, "double")},
    {"inj_signal.chirp_T",        make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::chirp_T, "double")},
    {"inj_signal.branch",         make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::branch, "string")},   // "LO|HO|ALL"
    {"inj_signal.apply_to",       make_nested_scalar_rw(&bdr_rtc_config::inj_signal, &bdr_signal_cfg::apply_to, "string")}, // "command|setpoint"
};


/*
We use a philosphy that more dynamic things like the controller (which can have different class implementations / rapidly varying values ) 
have list, get, set methods different to more static things in the rtc_config_struct. This is also due pointer impelmentation of the controller class.
*/

nlohmann::json list_rtc_fields() {
    nlohmann::json arr = nlohmann::json::array();

    for (const auto& kv : RTC_FIELDS) {
        arr.push_back({{"field", kv.first}, {"type", kv.second.type}});
    }
    return {{"ok", true}, {"fields", arr}};
}

// list dynamic controller parameters (can depend on what controller class you implement!)
nlohmann::json ctrl_list(nlohmann::json args) {
    using json = nlohmann::json;
    if (!args.is_array() || args.size() != 1) {
        return json{{"ok", false}, {"error","Expected [\"LO\"|\"HO\"]"}};
    }
    std::string which = args.at(0).get<std::string>();

    json arr = json::array();
    std::lock_guard<std::mutex> lk(ctrl_mutex);
    Controller* c = pick_ctrl(&rtc_config, which.c_str());
    if (!c) return json{{"ok", false}, {"error","controller not initialized"}, {"which", which}};

    // always show controller type
    arr.push_back({{"name","ctrl_type"}, {"type","string"}});

    for (const auto& [name, desc] : c->list_parameters()) {
        Controller::Param p;
        try { p = c->get_parameter(name); } catch (...) { continue; }
        const char* ty =
            std::holds_alternative<Eigen::VectorXd>(p) ? "vector<double>" :
            std::holds_alternative<Eigen::MatrixXd>(p) ? "matrix<double>" :
                                                         "unknown";
        arr.push_back({{"name",name}, {"type",ty}});
    }
    return json{{"ok", true}, {"which", which}, {"params", arr}};
}


nlohmann::json get_rtc_field(std::string path) {
    auto it = RTC_FIELDS.find(path);
    if (it == RTC_FIELDS.end()) {
        nlohmann::json keys = nlohmann::json::array();
        for (const auto& kv : RTC_FIELDS) keys.push_back(kv.first);
        return {{"ok", false}, {"error", "unknown field"}, {"field", path}, {"available", keys}};
    }
    const auto& fh = it->second;
    return {{"ok", true}, {"field", path}, {"type", fh.type}, {"value", fh.get()}};
}

// Usage: ctrl_get ["LO","kp"]      or  ctrl_get ["HO","ctrl_type"]
nlohmann::json ctrl_get(nlohmann::json args) {
    using json = nlohmann::json;
    if (!args.is_array() || args.size() != 2) {
        return json{{"ok", false}, {"error","Expected [\"LO\"|\"HO\", \"name\"]"}};
    }
    std::string which = args.at(0).get<std::string>();
    std::string name  = args.at(1).get<std::string>();

    std::lock_guard<std::mutex> lk(ctrl_mutex);
    Controller* c = pick_ctrl(&rtc_config, which.c_str());
    if (!c) return json{{"ok", false}, {"error","controller not initialized"}, {"which", which}};

    if (name == "ctrl_type") {
        return json{{"ok", true}, {"which", which}, {"name","ctrl_type"}, {"type","string"}, {"value", c->get_type()}};
    }

    try {
        Controller::Param p = c->get_parameter(name);
        if (std::holds_alternative<Eigen::VectorXd>(p)) {
            return json{{"ok", true}, {"which", which}, {"name", name}, {"type","vector<double>"},
                        {"value", eigen_vector_to_json(std::get<Eigen::VectorXd>(p))}};
        }
        if (std::holds_alternative<Eigen::MatrixXd>(p)) {
            return json{{"ok", true}, {"which", which}, {"name", name}, {"type","matrix<double>"},
                        {"value", eigen_matrix_to_json(std::get<Eigen::MatrixXd>(p))}};
        }
        return json{{"ok", false}, {"error","unsupported parameter kind"}, {"which", which}, {"name", name}};
    } catch (const std::exception& e) {
        json names = json::array();
        for (const auto& kv : c->list_parameters()) names.push_back(kv.first);
        return json{{"ok", false}, {"error", e.what()}, {"which", which}, {"name", name}, {"available", names}};
    }
}


nlohmann::json set_rtc_field(std::string path, nlohmann::json value) {
    auto it = RTC_FIELDS.find(path);
    if (it == RTC_FIELDS.end())
        return {{"ok", false}, {"error", "unknown field"}, {"field", path}};

    const auto& fh = it->second;
    if (!fh.set)
        return {{"ok", false}, {"error", "read-only field"}, {"field", path}};

    // Write the value
    const bool ok = fh.set(value);
    if (!ok)
        return {{"ok", false}, {"error", "bad value/type"}, {"field", path}, {"expected", fh.type}};

    // ---- Selective side-effects / derived recomputes ----
    // Matrices that affect runtime projections (cheap scale/update)
    if (path == "matrices.I2M_LO" || path == "scale") {
        std::lock_guard<std::mutex> lk(rtc_mutex);
        rtc_config.I2M_LO_runtime = rtc_config.matrices.I2M_LO;//rtc_config.scale * rtc_config.matrices.I2M_LO;
    }
    if (path == "matrices.I2M_HO" || path == "scale") {
        std::lock_guard<std::mutex> lk(rtc_mutex);
        rtc_config.I2M_HO_runtime = rtc_config.matrices.I2M_HO; //rtc_config.scale * rtc_config.matrices.I2M_HO;
    }
    // Strehl/rms model (m,b) recompute
    if (path == "matrices.I2rms_sec" || path == "scale") {
        std::lock_guard<std::mutex> lk(rtc_mutex);
        if (rtc_config.matrices.I2rms_sec.rows() >= 2 && rtc_config.matrices.I2rms_sec.cols() >= 2) {
            rtc_config.m_s_runtime = rtc_config.matrices.I2rms_sec(0,0); //rtc_config.scale * rtc_config.matrices.I2rms_sec(0,0);
            rtc_config.b_s_runtime =                       rtc_config.matrices.I2rms_sec(1,1);
        }
    }

    // Injection cache invalidation:
    // (basis→DM path or setpoint→DM mapping depends on these)
    if (path.rfind("inj_signal.", 0) == 0 ||
        path == "matrices.M2C_LO" ||
        path == "matrices.M2C_HO" ||
        path == "matrices.I2M_LO" ||
        path == "matrices.I2M_HO" ||
        path == "matrices.I2A")
    {
        mark_injection_changed();
    }

    // Echo back current value
    return {{"ok", true}, {"field", path}, {"type", fh.type}, {"value", fh.get()}};
}
// nlohmann::json set_rtc_field(std::string path, nlohmann::json value) {
//     auto it = RTC_FIELDS.find(path);
    
//     if (it == RTC_FIELDS.end())
//         return {{"ok", false}, {"error", "unknown field"}, {"field", path}};
//     const auto& fh = it->second;
//     if (!fh.set)
//         return {{"ok", false}, {"error", "read-only field"}, {"field", path}};
//     // Optional: per-field validation here (range/size checks, etc.)
//     const bool ok = fh.set(value);
//     if (!ok)
//         return {{"ok", false}, {"error", "bad value/type"}, {"field", path}, {"expected", fh.type}};
//     // Side-effects hook (recompute derived state) — add as needed, e.g.:
//     // if (path.rfind("ctrl_", 0) == 0) { std::lock_guard<std::mutex> lk(rtc_mutex); /* rebuild controllers */ }
//     if (path.rfind("inj_signal.", 0) == 0) {
//         mark_injection_changed();
//     }
//     return {{"ok", true}, {"field", path}, {"type", fh.type}, {"value", fh.get()}};
// }

// could be used to replace the set_ctrl_param function... 
// Usage: ctrl_set ["LO","kp", [0.1,0.1,...]]
//        ctrl_set ["HO","upper_limits", [2,2,...]]
//        ctrl_set ["LO","M", [[...], [...]]]   // if a controller exposes a matrix param
nlohmann::json ctrl_set(nlohmann::json args) {
    using json = nlohmann::json;
    if (!args.is_array() || args.size() != 3) {
        return json{{"ok", false}, {"error","Expected [\"LO\"|\"HO\", \"name\", value]"}};
    }
    std::string which = args.at(0).get<std::string>();
    std::string name  = args.at(1).get<std::string>();
    const json  value = args.at(2);

    if (name == "ctrl_type") {
        return json{{"ok", false}, {"error","ctrl_type is read-only"}, {"which", which}};
    }

    // We'll fill these and return after releasing the lock:
    std::string type  = "unknown";
    json        echo  = json::object();

    {
        std::lock_guard<std::mutex> lk(ctrl_mutex);
        Controller* c = pick_ctrl(&rtc_config, which.c_str());
        if (!c) return json{{"ok", false}, {"error","controller not initialized"}, {"which", which}};

        // vector?
        Eigen::VectorXd v;
        if (json_to_eigen_vector(value, v)) {
            c->set_parameter(name, v);
            // fetch back to echo
            Controller::Param p = c->get_parameter(name);
            type = "vector<double>";
            echo = eigen_vector_to_json(std::get<Eigen::VectorXd>(p));
        }
        // matrix?
        else {
            Eigen::MatrixXd M;
            if (json_to_eigen_matrix(value, M)) {
                c->set_parameter(name, M);
                Controller::Param p = c->get_parameter(name);
                type = "matrix<double>";
                echo = eigen_matrix_to_json(std::get<Eigen::MatrixXd>(p));
            } else {
                return json{{"ok", false}, {"error","value is neither vector nor matrix"},
                            {"which", which}, {"name", name}};
            }
        }
    } // lock released here

    return json{{"ok", true}, {"which", which}, {"name", name}, {"type", type}, {"value", echo}};
}



//// working version 1 
// //// rtc config file getter and setters ! 
// // // test in TTR115.0035 (not tested yet)

// // A handler for a single exposed field on rtc_config.
// // We only USE .get() now; .set() is reserved for later.
// struct FieldHandle {
//     std::string type;                             // "double", "int", "bool", "string", "vector<double>", ...
//     std::function<nlohmann::json()> get;          // returns current value as JSON
//     std::function<bool(const nlohmann::json&)> set;  // optional; unused for now
// };

// // --- Helper factories: GETTER ONLY now; set = nullptr for expansion later ---

// template<typename T>
// static FieldHandle make_scalar_field_getter(T bdr_rtc_config::* ptr, const char* type_name) {
//     return FieldHandle{
//         type_name,
//         // getter
//         [ptr]() -> nlohmann::json {
//             std::lock_guard<std::mutex> lk(rtc_mutex);
//             return rtc_config.*ptr;               // nlohmann::json supports numbers/bools/strings directly
//         },
//         /*set*/ nullptr
//     };
// }

// template<typename Sub, typename T>
// static FieldHandle make_nested_scalar_getter(Sub bdr_rtc_config::* sub, T Sub::* mem, const char* type_name) {
//     return FieldHandle{
//         type_name,
//         [sub, mem]() -> nlohmann::json {
//             std::lock_guard<std::mutex> lk(rtc_mutex);
//             return (rtc_config.*sub).*mem;
//         },
//         /*set*/ nullptr
//     };
// }

// template<typename T>
// static FieldHandle make_vector_field_getter(std::vector<T> bdr_rtc_config::* ptr, const char* type_name) {
//     return FieldHandle{
//         type_name,
//         [ptr]() -> nlohmann::json {
//             std::lock_guard<std::mutex> lk(rtc_mutex);
//             return rtc_config.*ptr;               // std::vector<T> -> JSON array
//         },
//         /*set*/ nullptr
//     };
// }

// // Helper for nested Eigen::VectorXd -> JSON array
// template<typename Sub>
// static FieldHandle make_nested_eigenvec_getter(Sub bdr_rtc_config::* sub,
//                                                Eigen::VectorXd Sub::* mem,
//                                                const char* type_name = "vector<double>") {
//     return FieldHandle{
//         type_name,
//         [sub, mem]() -> nlohmann::json {
//             std::lock_guard<std::mutex> lk(rtc_mutex);
//             const Eigen::VectorXd& v = (rtc_config.*sub).*mem;
//             return std::vector<double>(v.data(), v.data() + v.size());
//         },
//         /*set*/ nullptr
//     };
// }

// // ---------- Generic Eigen -> JSON helpers (vectors & matrices) ----------
// template<typename EigenVec>
// static nlohmann::json eigen_vector_to_json(const EigenVec& v) {
//     using S = typename EigenVec::Scalar;
//     // Avoid char-like JSON by widening integral scalars
//     if constexpr (std::is_integral_v<S>) {
//         std::vector<long long> out(v.size());
//         for (Eigen::Index i = 0; i < v.size(); ++i) out[i] = static_cast<long long>(v(i));
//         return out;
//     } else {
//         std::vector<double> out(v.size());
//         for (Eigen::Index i = 0; i < v.size(); ++i) out[i] = static_cast<double>(v(i));
//         return out;
//     }
// }

// template<typename EigenMat>
// static nlohmann::json eigen_matrix_to_json(const EigenMat& M) {
//     using S = typename EigenMat::Scalar;
//     nlohmann::json rows = nlohmann::json::array();
//     for (Eigen::Index r = 0; r < M.rows(); ++r) {
//         if constexpr (std::is_integral_v<S>) {
//             std::vector<long long> row(M.cols());
//             for (Eigen::Index c = 0; c < M.cols(); ++c) row[c] = static_cast<long long>(M(r,c));
//             rows.push_back(row);
//         } else {
//             std::vector<double> row(M.cols());
//             for (Eigen::Index c = 0; c < M.cols(); ++c) row[c] = static_cast<double>(M(r,c));
//             rows.push_back(row);
//         }
//     }
//     return rows;
// }

// // ---------- FieldHandle factories for Eigen (nested and top-level) ----------
// template<typename Sub, typename EigenVec>
// static FieldHandle make_nested_eigen_vector_getter(Sub bdr_rtc_config::* sub,
//                                                    EigenVec Sub::* mem,
//                                                    const char* type_name = "vector") {
//     return FieldHandle{
//         type_name,
//         [sub, mem]() -> nlohmann::json {
//             std::lock_guard<std::mutex> lk(rtc_mutex);
//             return eigen_vector_to_json((rtc_config.*sub).*mem);
//         },
//         /*set*/ nullptr
//     };
// }

// template<typename Sub, typename EigenMat>
// static FieldHandle make_nested_eigen_matrix_getter(Sub bdr_rtc_config::* sub,
//                                                    EigenMat Sub::* mem,
//                                                    const char* type_name = "matrix") {
//     return FieldHandle{
//         type_name,
//         [sub, mem]() -> nlohmann::json {
//             std::lock_guard<std::mutex> lk(rtc_mutex);
//             return eigen_matrix_to_json((rtc_config.*sub).*mem);
//         },
//         /*set*/ nullptr
//     };
// }

// template<typename EigenVec>
// static FieldHandle make_eigen_vector_field_getter(EigenVec bdr_rtc_config::* mem,
//                                                   const char* type_name = "vector") {
//     return FieldHandle{
//         type_name,
//         [mem]() -> nlohmann::json {
//             std::lock_guard<std::mutex> lk(rtc_mutex);
//             return eigen_vector_to_json(rtc_config.*mem);
//         },
//         /*set*/ nullptr
//     };
// }

// template<typename EigenMat>
// static FieldHandle make_eigen_matrix_field_getter(EigenMat bdr_rtc_config::* mem,
//                                                   const char* type_name = "matrix") {
//     return FieldHandle{
//         type_name,
//         [mem]() -> nlohmann::json {
//             std::lock_guard<std::mutex> lk(rtc_mutex);
//             return eigen_matrix_to_json(rtc_config.*mem);
//         },
//         /*set*/ nullptr
//     };
// }

// // // ---------------- Register ONLY the fields we want visible ----------------
// // // Start empty; add lines like the commented examples below when ready.
// // // NOTE: Update names to match your actual bdr_rtc_config definition.
// // static const std::unordered_map<std::string, FieldHandle> RTC_FIELDS = {
// //     // Example top-level scalars:
// //     // {"fps_override",      make_scalar_field_getter(&bdr_rtc_config::fps_override, "double")},
// //     // {"gain_override",     make_scalar_field_getter(&bdr_rtc_config::gain_override, "double")},
// //     // {"override_gain_fps", make_scalar_field_getter(&bdr_rtc_config::override_gain_fps, "bool")},

// //     // Example nested (replace 'CtrlType' and members with your real types):
// //     // {"ctrl_LO.kp", make_nested_scalar_getter(&bdr_rtc_config::ctrl_LO, &CtrlType::kp, "double")},
// //     { "ctrl_LO.ki",make_nested_eigenvec_getter(&bdr_rtc_config::ctrl_LO, &PIDController::ki) },
// //     // {"ctrl_HO.rho", make_nested_scalar_getter(&bdr_rtc_config::ctrl_HO, &LeakyIntegrator::rho, "double")},

// //     // Example vector:
// //     // {"modal_gains_LO", make_vector_field_getter(&bdr_rtc_config::modal_gains_LO, "vector<double>")},
// // };
// static const std::unordered_map<std::string, FieldHandle> RTC_FIELDS = {
//     // ----- state -----
//     {"state.DM_flat",            make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::DM_flat, "string")},
//     {"state.signal_space",       make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::signal_space, "string")},
//     {"state.LO",                 make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::LO, "int")},
//     {"state.controller_type",    make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::controller_type, "string")},
//     {"state.inverse_method_LO",  make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::inverse_method_LO, "string")},
//     {"state.inverse_method_HO",  make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::inverse_method_HO, "string")},
//     {"state.phasemask",          make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::phasemask, "string")},
//     {"state.auto_close",         make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::auto_close, "int")},
//     {"state.auto_open",          make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::auto_open, "int")},
//     {"state.auto_tune",          make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::auto_tune, "int")},
//     {"state.take_telemetry",     make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::take_telemetry, "int")},
//     {"state.simulation_mode",    make_nested_scalar_getter(&bdr_rtc_config::state, &bdr_state::simulation_mode, "int")},

//     // ----- reduction (Eigen::VectorXd) -----
//     {"reduction.bias",           make_nested_eigen_vector_getter(&bdr_rtc_config::reduction, &bdr_reduction::bias, "vector<double>")},
//     {"reduction.bias_dm",        make_nested_eigen_vector_getter(&bdr_rtc_config::reduction, &bdr_reduction::bias_dm, "vector<double>")},
//     {"reduction.dark",           make_nested_eigen_vector_getter(&bdr_rtc_config::reduction, &bdr_reduction::dark, "vector<double>")},
//     {"reduction.dark_dm",        make_nested_eigen_vector_getter(&bdr_rtc_config::reduction, &bdr_reduction::dark_dm, "vector<double>")},

//     // ----- pixels (integer Eigen vectors) -----
//     {"pixels.crop_pixels",       make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::crop_pixels, "vector<int16>")},
//     {"pixels.bad_pixels",        make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::bad_pixels, "vector<int32>")},
//     {"pixels.pupil_pixels",      make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::pupil_pixels, "vector<int32>")},
//     {"pixels.interior_pixels",   make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::interior_pixels, "vector<int32>")},
//     {"pixels.secondary_pixels",  make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::secondary_pixels, "vector<int32>")},
//     {"pixels.exterior_pixels",   make_nested_eigen_vector_getter(&bdr_rtc_config::pixels, &bdr_pixels::exterior_pixels, "vector<int32>")},

//     // ----- reference_pupils (Eigen::VectorXd) -----
//     {"reference_pupils.I0",           make_nested_eigen_vector_getter(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::I0, "vector<double>")},
//     {"reference_pupils.N0",           make_nested_eigen_vector_getter(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::N0, "vector<double>")},
//     {"reference_pupils.norm_pupil",   make_nested_eigen_vector_getter(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::norm_pupil, "vector<double>")},
//     {"reference_pupils.norm_pupil_dm",make_nested_eigen_vector_getter(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::norm_pupil_dm, "vector<double>")},
//     {"reference_pupils.I0_dm",        make_nested_eigen_vector_getter(&bdr_rtc_config::reference_pupils, &bdr_refence_pupils::I0_dm, "vector<double>")},

//     // ----- matrices (Eigen::MatrixXd) + sizes -----
//     {"matrices.I2A",        make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2A, "matrix<double>")},
//     {"matrices.I2M",        make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2M, "matrix<double>")},
//     {"matrices.I2M_LO",     make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2M_LO, "matrix<double>")},
//     {"matrices.I2M_HO",     make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2M_HO, "matrix<double>")},
//     {"matrices.M2C",        make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::M2C, "matrix<double>")},
//     {"matrices.M2C_LO",     make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::M2C_LO, "matrix<double>")},
//     {"matrices.M2C_HO",     make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::M2C_HO, "matrix<double>")},
//     {"matrices.I2rms_sec",  make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2rms_sec, "matrix<double>")},
//     {"matrices.I2rms_ext",  make_nested_eigen_matrix_getter(&bdr_rtc_config::matrices, &bdr_matricies::I2rms_ext, "matrix<double>")},
//     {"matrices.szm",        make_nested_scalar_getter(&bdr_rtc_config::matrices, &bdr_matricies::szm, "int")},
//     {"matrices.sza",        make_nested_scalar_getter(&bdr_rtc_config::matrices, &bdr_matricies::sza, "int")},
//     {"matrices.szp",        make_nested_scalar_getter(&bdr_rtc_config::matrices, &bdr_matricies::szp, "int")},

//     // ----- LO/HO controller configs (bdr_controller) -----
//     {"ctrl_LO_config.kp",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::kp, "vector<double>")},
//     {"ctrl_LO_config.ki",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::ki, "vector<double>")},
//     {"ctrl_LO_config.kd",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::kd, "vector<double>")},
//     {"ctrl_LO_config.lower_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::lower_limits, "vector<double>")},
//     {"ctrl_LO_config.upper_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::upper_limits, "vector<double>")},
//     {"ctrl_LO_config.set_point",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO_config, &bdr_controller::set_point, "vector<double>")},

//     {"ctrl_HO_config.kp",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::kp, "vector<double>")},
//     {"ctrl_HO_config.ki",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::ki, "vector<double>")},
//     {"ctrl_HO_config.kd",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::kd, "vector<double>")},
//     {"ctrl_HO_config.lower_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::lower_limits, "vector<double>")},
//     {"ctrl_HO_config.upper_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::upper_limits, "vector<double>")},
//     {"ctrl_HO_config.set_point",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO_config, &bdr_controller::set_point, "vector<double>")},

//     // ----- loop limits -----
//     {"limits.close_on_strehl_limit", make_nested_scalar_getter(&bdr_rtc_config::limits, &bdr_limits::close_on_strehl_limit, "float")},
//     {"limits.open_on_strehl_limit",  make_nested_scalar_getter(&bdr_rtc_config::limits, &bdr_limits::open_on_strehl_limit, "float")},
//     {"limits.open_on_flux_limit",    make_nested_scalar_getter(&bdr_rtc_config::limits, &bdr_limits::open_on_flux_limit, "float")},
//     {"limits.open_on_dm_limit",      make_nested_scalar_getter(&bdr_rtc_config::limits, &bdr_limits::open_on_dm_limit, "float")},
//     {"limits.LO_offload_limit",      make_nested_scalar_getter(&bdr_rtc_config::limits, &bdr_limits::LO_offload_limit, "float")},

//     // ----- camera (strings) -----
//     {"cam.fps",               make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::fps, "string")},
//     {"cam.gain",              make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::gain, "string")},
//     {"cam.testpattern",       make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::testpattern, "string")},
//     {"cam.bias",              make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::bias, "string")},
//     {"cam.flat",              make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::flat, "string")},
//     {"cam.imagetags",         make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::imagetags, "string")},
//     {"cam.led",               make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::led, "string")},
//     {"cam.events",            make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::events, "string")},
//     {"cam.extsynchro",        make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::extsynchro, "string")},
//     {"cam.rawimages",         make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::rawimages, "string")},
//     {"cam.cooling",           make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::cooling, "string")},
//     {"cam.mode",              make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::mode, "string")},
//     {"cam.resetwidth",        make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::resetwidth, "string")},
//     {"cam.nbreadworeset",     make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::nbreadworeset, "string")},
//     {"cam.cropping",          make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::cropping, "string")},
//     {"cam.cropping_columns",  make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::cropping_columns, "string")},
//     {"cam.cropping_rows",     make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::cropping_rows, "string")},
//     {"cam.aduoffset",         make_nested_scalar_getter(&bdr_rtc_config::cam, &bdr_cam::aduoffset, "string")},

//     // ----- filters (uint8 masks + VectorXf DM projections) -----
//     {"filters.bad_pixel_mask",   make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::bad_pixel_mask, "vector<uint8>")},
//     {"filters.bad_pixel_mask_dm",make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::bad_pixel_mask_dm, "vector<float>")},
//     {"filters.pupil",            make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::pupil, "vector<uint8>")},
//     {"filters.pupil_dm",         make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::pupil_dm, "vector<float>")},
//     {"filters.secondary",        make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::secondary, "vector<uint8>")},
//     {"filters.secondary_dm",     make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::secondary_dm, "vector<float>")},
//     {"filters.exterior",         make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::exterior, "vector<uint8>")},
//     {"filters.exterior_dm",      make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::exterior_dm, "vector<float>")},
//     {"filters.inner_pupil_filt", make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::inner_pupil_filt, "vector<uint8>")},
//     {"filters.inner_pupil_filt_dm", make_nested_eigen_vector_getter(&bdr_rtc_config::filters, &bdr_filters::inner_pupil_filt_dm, "vector<float>")},

//     // ================= Derived/runtime (top-level) =================
//     {"zeroCmd",              make_eigen_vector_field_getter(&bdr_rtc_config::zeroCmd, "vector<double>")},
//     {"fps",                  make_scalar_field_getter(&bdr_rtc_config::fps, "double")},
//     {"gain",                 make_scalar_field_getter(&bdr_rtc_config::gain, "double")},
//     {"scale",                make_scalar_field_getter(&bdr_rtc_config::scale, "double")},
//     {"I2M_LO_runtime",       make_eigen_matrix_field_getter(&bdr_rtc_config::I2M_LO_runtime, "matrix<double>")},
//     {"I2M_HO_runtime",       make_eigen_matrix_field_getter(&bdr_rtc_config::I2M_HO_runtime, "matrix<double>")},
//     {"N0_dm_runtime",        make_eigen_vector_field_getter(&bdr_rtc_config::N0_dm_runtime, "vector<double>")},
//     {"I0_dm_runtime",        make_eigen_vector_field_getter(&bdr_rtc_config::I0_dm_runtime, "vector<double>")},
//     {"dark_dm_runtime",      make_eigen_vector_field_getter(&bdr_rtc_config::dark_dm_runtime, "vector<double>")},
//     {"sec_idx",              make_scalar_field_getter(&bdr_rtc_config::sec_idx, "int")},
//     {"m_s_runtime",          make_scalar_field_getter(&bdr_rtc_config::m_s_runtime, "double")},
//     {"b_s_runtime",          make_scalar_field_getter(&bdr_rtc_config::b_s_runtime, "double")},

//     // ----- runtime controllers (PIDController) -----
//     {"ctrl_LO.ctrl_type",    make_nested_scalar_getter(&bdr_rtc_config::ctrl_LO, &PIDController::ctrl_type, "string")},
//     {"ctrl_LO.kp",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::kp, "vector<double>")},
//     {"ctrl_LO.ki",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::ki, "vector<double>")},
//     {"ctrl_LO.kd",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::kd, "vector<double>")},
//     {"ctrl_LO.lower_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::lower_limits, "vector<double>")},
//     {"ctrl_LO.upper_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::upper_limits, "vector<double>")},
//     {"ctrl_LO.set_point",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::set_point, "vector<double>")},
//     {"ctrl_LO.output",       make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::output, "vector<double>")},
//     {"ctrl_LO.integrals",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::integrals, "vector<double>")},
//     {"ctrl_LO.prev_errors",  make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_LO, &PIDController::prev_errors, "vector<double>")},

//     {"ctrl_HO.ctrl_type",    make_nested_scalar_getter(&bdr_rtc_config::ctrl_HO, &PIDController::ctrl_type, "string")},
//     {"ctrl_HO.kp",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::kp, "vector<double>")},
//     {"ctrl_HO.ki",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::ki, "vector<double>")},
//     {"ctrl_HO.kd",           make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::kd, "vector<double>")},
//     {"ctrl_HO.lower_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::lower_limits, "vector<double>")},
//     {"ctrl_HO.upper_limits", make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::upper_limits, "vector<double>")},
//     {"ctrl_HO.set_point",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::set_point, "vector<double>")},
//     {"ctrl_HO.output",       make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::output, "vector<double>")},
//     {"ctrl_HO.integrals",    make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::integrals, "vector<double>")},
//     {"ctrl_HO.prev_errors",  make_nested_eigen_vector_getter(&bdr_rtc_config::ctrl_HO, &PIDController::prev_errors, "vector<double>")}
//     // --- NOTE: telemetry ring-buffers (telem.*) deliberately not exposed here ---
// };
// // --------------- Commander handlers: list + get (getter-only) ---------------

// nlohmann::json list_rtc_fields() {
//     nlohmann::json arr = nlohmann::json::array();
//     for (const auto& kv : RTC_FIELDS) {
//         arr.push_back({{"field", kv.first}, {"type", kv.second.type}});
//     }
//     return {{"ok", true}, {"fields", arr}};
// }

// // GET one field by "path" (e.g. "fps_override" or "ctrl_LO.kp")
// nlohmann::json get_rtc_field(std::string path) {
//     auto it = RTC_FIELDS.find(path);
//     if (it == RTC_FIELDS.end()) {
//         nlohmann::json keys = nlohmann::json::array();
//         for (const auto& kv : RTC_FIELDS) keys.push_back(kv.first);
//         return {{"ok", false}, {"error", "unknown field"}, {"field", path}, {"available", keys}};
//     }
//     const auto& fh = it->second;
//     return {{"ok", true}, {"field", path}, {"type", fh.type}, {"value", fh.get()}};
// }

// //// END rtc config file getter and setters ! 








void stop_baldr() {
    // mode 2 is stoppoing
    servo_mode = SERVO_STOP;
    std::cout << "servo_mode updated to " << servo_mode << "stopping Baldr RTC" << std::endl;
}

void close_baldr_LO() {
    // mode 1 is close
    servo_mode_LO = SERVO_CLOSE;
    std::cout << "servo_mode_LO updated to " << servo_mode_LO << ". CLosing Baldr LO modes" << std::endl;
}

void close_all() {
    // mode 1 is close
    servo_mode_LO = SERVO_CLOSE;
    std::cout << "servo_mode_LO updated to " << servo_mode_LO << ". CLosing Baldr LO modes" << std::endl;
    servo_mode_HO = SERVO_CLOSE;
    std::cout << "servo_mode_HO updated to " << servo_mode_LO << ". CLosing Baldr HO modes" << std::endl;
}

void open_all() {
    // mode 1 is close
    servo_mode_LO = SERVO_OPEN;
    servo_mode_HO = SERVO_OPEN;
    std::cout << "servo_mode_LO updated to " << servo_mode_LO << ". CLosing Baldr LO modes" << std::endl;
}

void open_baldr_LO() {
    // mode 0 is open
    servo_mode_LO = SERVO_OPEN;
    std::cout << "servo_mode_LO updated to  " << servo_mode_LO << ". Opening Baldr LO modes" << std::endl;
}

void close_baldr_HO() {
    // mode 1 is close
    servo_mode_HO = SERVO_CLOSE;
    std::cout << "servo_mode_HO updated to " << servo_mode_HO << ". CLosing Baldr HO modes" << std::endl;
}

void open_baldr_HO() {
    // mode 0 is open
    servo_mode_HO = SERVO_OPEN;
    std::cout << "servo_mode_HO updated to  " << servo_mode_HO << ". Opening Baldr HO modes" << std::endl;
}


//// original wrong norm
// void I0_update() {
//     // !!! only updates rtc_config.I0_dm_runtime !!! 
//     // does not update rtc_config.reference_pupils members 
//     // assumes we are in a correct state to obtain reference ZWFS pupil - we do not checks 
//     if (rtc_config.telem.img_dm.empty()) {
//         std::cerr << "[I0_update] Warning: telemetry buffer is empty, cannot update I0." << std::endl;
//         return;
//     }

//     // Initialize a sum vector with zeros, same size as first vector in img_dm
//     Eigen::VectorXd sum = Eigen::VectorXd::Zero(rtc_config.telem.img_dm.front().size());

//     size_t count = 0;
//     // read from the telem ring buffer
//     for (const auto& v : rtc_config.telem.img_dm) {
//         if (v.size() != sum.size()) {
//             std::cerr << "[I0_update] Warning: found img_dm vector with inconsistent size, skipping." << std::endl;
//             continue; // Skip badly-sized vector
//         }
//         sum += v;
//         ++count;
//     }

//     if (count == 0) {
//         std::cerr << "[I0_update] Error: no valid vectors found for averaging." << std::endl;
//         return;
//     }

//     // Compute average
//     rtc_config.I0_dm_runtime = sum / static_cast<double>(count);

//     // Subtraction of dark and bias was previously done in img_dm calc held in rtc telem ring buffer !!! 
//     // but since Paranal AIV darks are subtracted in camera server so this internal subtraction is no longer used 

//     std::cout << "[I0_update] I0_dm_runtime updated from " << count << " samples." << std::endl;
// }


// Generic temporal mean over any iterable container of Eigen::VectorXd
// (works with boost::circular_buffer, std::vector, etc.)
template <typename Container>
static bool temporal_mean(const Container& buf, Eigen::VectorXd& mean_out) {
    if (buf.empty()) return false;
    const auto L = buf.front().size();
    if (L == 0) return false;

    mean_out = Eigen::VectorXd::Zero(L);
    size_t n = 0;
    for (const auto& v : buf) {
        if (v.size() != L) {
            std::cerr << "[temporal_mean] Warning: vector size mismatch (" << v.size()
                      << " vs " << L << "), skipping.\n";
            continue;
        }
        mean_out.noalias() += v;
        ++n;
    }
    if (n == 0) return false;
    mean_out.array() /= static_cast<double>(n);
    return true;
}

// original errounous version that doesnt consistently update intrn_flx_I0!!
// void I0_update() {
//     std::lock_guard<std::mutex> lock(telemetry_mutex);

//     Eigen::VectorXd mean_dm;   // <rtc_config.telem.img_dm>, already per-frame normalized in the loop
//     if (!temporal_mean(rtc_config.telem.img_dm, mean_dm)) {
//         std::cerr << "[I0_update] Error: cannot form temporal mean of img_dm.\n";
//         return;
//     }

//     // Runtime projected reference in *same* normalization as img_dm
//     rtc_config.I0_dm_runtime = mean_dm; // in the rtc we already normalize the img_dm by the sum(subframe) so no need to repeat here ! we just update with the average telem.img_dm image  

//     // I also need to update the rtc_config.reference_pupils.intrn_flx_I0 here to keep it consistent with the new I0_dm_runtime
    
//     // Psuedo code I need to finish here
//     raw_imgs = temporal_mean(rtc_config.telem.img, imgs_tmp);
//     ref_imgs_red = raw_imgs - rtc_config.reduction.dark; // subtract dark from subframe (size 32x32)
//     rtc_config.reference_pupils.intrn_flx_I0 = ref_imgs_red.sum(); // sum of the mean image in the telem buffer
//     // end pseudo code

//     // Intentionally do NOT touch rtc_config.reference_pupils.I0 here to avoid mixing spaces.
//     std::cout << "[I0_update] Updated I0_dm_runtime (len=" << mean_dm.size() << ").\n";
// }

void I0_update() {
    std::lock_guard<std::mutex> lock(telemetry_mutex);

    // I0 in DM-space from telemetry (already normalized in rtc loop)
    Eigen::VectorXd mean_dm;   // average of telem.img_dm
    if (!temporal_mean(rtc_config.telem.img_dm, mean_dm)) {
        std::cerr << "[I0_update] Error: cannot form temporal mean of img_dm.\n";
        return;
    }
    rtc_config.I0_dm_runtime = mean_dm;  // same normalization as runtime img_dm

    // Update intrn_flx_I0 to keep RTC scaling consistent with current I0
    //    This must mirror rtc.cpp: img_red = img - reduction.dark; g_subframe_int = img_red.sum()
    Eigen::VectorXd mean_img;  // average of RAW telem.img (per-frame ADU)
    if (!temporal_mean(rtc_config.telem.img, mean_img)) {
        std::cerr << "[I0_update] Warning: cannot form temporal mean of RAW img; "
                     "leaving intrn_flx_I0 unchanged.\n";
    } else {
        // Ensure dark is initialized & sized; default to flat 1000 ADU if needed
        if (rtc_config.reduction.dark.size() != mean_img.size()) {
            rtc_config.reduction.dark = Eigen::VectorXd::Constant(mean_img.size(), 1000.0);
        }

        const Eigen::VectorXd mean_img_red = mean_img - rtc_config.reduction.dark;
        const double sub_int = mean_img_red.sum();  // ADU per frame, matches g_subframe_int units

        if (std::isfinite(sub_int) && sub_int > 0.0) {
            rtc_config.reference_pupils.intrn_flx_I0 = sub_int;
        } else {
            std::cerr << "[I0_update] Warning: invalid intrn_flx_I0 candidate (" << sub_int
                      << "); leaving previous value.\n";
        }
    }

    std::cout << "[I0_update] Updated I0_dm_runtime (len=" << mean_dm.size()
              << "); intrn_flx_I0=" << rtc_config.reference_pupils.intrn_flx_I0 << "\n";
}




// Lucky I0 update: use only the top-Nth percentile frames by dm_rms_est_1
// percentile: e.g. 90.0 means "use frames >= 90th percentile of dm_rms_est_1"
// min_samples: require at least this many frames in the telemetry buffers to proceed

void I0_update_lucky(double percentile = 90.0, int min_samples = 10) {
    std::lock_guard<std::mutex> lock(telemetry_mutex);

    // Shorthand refs (assumed index-aligned)
    const auto& buf_dm   = rtc_config.telem.img_dm;        // vector<Eigen::VectorXd>
    const auto& buf_snr  = rtc_config.telem.rmse_est;      // vector<double> (or similar)
    const auto& buf_img  = rtc_config.telem.img;           // vector<Eigen::VectorXd> (RAW ADU)

    const size_t K_dm  = buf_dm.size();
    const size_t K_snr = buf_snr.size();
    const size_t K_img = buf_img.size();
    const size_t K     = std::min(K_dm, K_snr);

    if (K < static_cast<size_t>(min_samples)) {
        std::cerr << "[I0_update_lucky] Not enough telemetry: have " << K
                  << " frames (min " << min_samples << ").\n";
        return;
    }

    // Build metrics array
    std::vector<double> metrics;
    metrics.reserve(K);
    for (size_t i = 0; i < K; ++i) {
        metrics.push_back(static_cast<double>(buf_snr[i]));
    }

    // Clamp percentile (keep enough frames)
    double p = std::max(50.0, std::min(99.9, percentile));
    const size_t nth_idx = static_cast<size_t>(std::ceil((p / 100.0) * (K - 1)));

    // Threshold at p-th percentile
    std::vector<double> work = metrics;
    std::nth_element(work.begin(), work.begin() + nth_idx, work.end());
    const double thresh = work[nth_idx];

    // Accumulators for selected frames
    Eigen::VectorXd acc_dm;
    Eigen::VectorXd acc_img;   // RAW frames (only if K_img is aligned/available)
    Eigen::Index P_dm = 0, P_img = 0;
    size_t kept = 0;

    const bool can_update_intrn = (K_img >= K);  // require at least K raw frames for aligned indexing

    for (size_t i = 0; i < K; ++i) {
        if (metrics[i] >= thresh) {
            // DM-space frame
            const Eigen::VectorXd& fdm = buf_dm[i];
            if (P_dm == 0) {
                P_dm = fdm.size();
                if (P_dm == 0) {
                    std::cerr << "[I0_update_lucky] First selected img_dm frame is empty.\n";
                    return;
                }
                acc_dm = Eigen::VectorXd::Zero(P_dm);
            }
            if (fdm.size() == P_dm) acc_dm += fdm;  // skip mismatched silently

            // RAW frame (optional, for intrn update)
            if (can_update_intrn) {
                const Eigen::VectorXd& fimg = buf_img[i];
                if (P_img == 0) {
                    P_img = fimg.size();
                    if (P_img == 0) {
                        std::cerr << "[I0_update_lucky] First selected RAW img frame is empty.\n";
                        // we can still proceed for I0_dm; just disable intrn update
                    } else {
                        acc_img = Eigen::VectorXd::Zero(P_img);
                    }
                }
                if (P_img > 0 && fimg.size() == P_img) acc_img += fimg;
            }

            ++kept;
        }
    }

    if (kept == 0 || P_dm == 0) {
        std::cerr << "[I0_update_lucky] No valid frames met the " << p
                  << "th percentile (thresh=" << thresh << ").\n";
        return;
    }

    // Update I0_dm_runtime from selected frames
    rtc_config.I0_dm_runtime = acc_dm / static_cast<double>(kept);

    // Update intrn_flx_I0 from selected RAW frames, if available
    if (P_img > 0 && acc_img.size() == P_img) {
        Eigen::VectorXd mean_img = acc_img / static_cast<double>(kept);

        // Ensure dark vector is sized; lazy default to 1000 ADU if needed
        if (rtc_config.reduction.dark.size() != mean_img.size()) {
            std::cerr << "[I0_update_lucky] Warning: reduction.dark is not same size as image. Using lazy 1000 ADU dark value (the default adu offset) " << std::endl;
            rtc_config.reduction.dark = Eigen::VectorXd::Constant(mean_img.size(), 1000.0);
        }

        const Eigen::VectorXd mean_img_red = mean_img - rtc_config.reduction.dark;
        const double sub_int = mean_img_red.sum();  // matches g_subframe_int units (ADU per frame)

        if (std::isfinite(sub_int) && sub_int > 0.0) {
            rtc_config.reference_pupils.intrn_flx_I0 = sub_int;
        } else {
            std::cerr << "[I0_update_lucky] Warning: invalid intrn_flx_I0 candidate (" << sub_int
                      << "); leaving previous value.\n";
        }
    } else {
        std::cerr << "[I0_update_lucky] Note: RAW img telemetry unavailable/mismatched; "
                     "I0_dm_runtime updated, intrn_flx_I0 left unchanged.\n";
    }

    std::cout << "[I0_update_lucky] Updated I0_dm_runtime using top " << p
              << "th percentile (" << kept << "/" << K << " frames). "
              << "intrn_flx_I0=" << rtc_config.reference_pupils.intrn_flx_I0 << "\n";
}

// // errornous version that doesnt consistently update intrn_flx_I0!!
// void I0_update_lucky(double percentile = 90.0, int min_samples = 10) {
//     std::lock_guard<std::mutex> lock(telemetry_mutex);

//     // Shorthand refs
//     const auto& buf_dm  = rtc_config.telem.img_dm;        // container of Eigen::VectorXd
//     const auto& buf_snr = rtc_config.telem.dm_rms_est_1;  // container of double (or 1D Eigen)

//     const size_t K_dm  = buf_dm.size();
//     const size_t K_snr = buf_snr.size();
//     const size_t K     = std::min(K_dm, K_snr);

//     if (K < static_cast<size_t>(min_samples)) {
//         std::cerr << "[I0_update_lucky] Not enough telemetry: have " << K
//                   << " frames, need >= " << min_samples << ".\n";
//         return;
//     }

//     // Copy metrics to a plain vector<double> so we can nth_element
//     std::vector<double> metrics;
//     metrics.reserve(K);
//     for (size_t i = 0; i < K; ++i) {
//         // If buf_snr is Eigen::VectorXd per-frame (1 element), adapt here; otherwise assume double.
//         metrics.push_back(static_cast<double>(buf_snr[i]));
//     }

//     // Clamp percentile to a reasonable range (avoid empty selections)
//     double p = std::max(50.0, std::min(99.9, percentile));  // default 90.0 is fine
//     const size_t nth_idx = static_cast<size_t>(std::ceil((p / 100.0) * (K - 1)));

//     // Find threshold at the p-th percentile
//     std::vector<double> work = metrics;  // nth_element works in-place
//     std::nth_element(work.begin(), work.begin() + nth_idx, work.end());
//     const double thresh = work[nth_idx];

//     // Accumulate selected frames
//     Eigen::VectorXd acc;
//     Eigen::Index P = 0;
//     size_t kept = 0;

//     for (size_t i = 0; i < K; ++i) {
//         if (metrics[i] >= thresh) {
//             const Eigen::VectorXd& f = buf_dm[i];
//             if (P == 0) {
//                 P = f.size();
//                 if (P == 0) {
//                     std::cerr << "[I0_update_lucky] First selected frame is empty.\n";
//                     return;
//                 }
//                 acc = Eigen::VectorXd::Zero(P);
//             }
//             if (f.size() != P) {
//                 // Skip inconsistent frames silently; they shouldn't occur
//                 continue;
//             }
//             acc += f;
//             ++kept;
//         }
//     }

//     if (kept == 0) {
//         std::cerr << "[I0_update_lucky] No frames met the " << p << "th percentile (thresh="
//                   << thresh << ").\n";
//         return;
//     }

//     rtc_config.I0_dm_runtime = acc / static_cast<double>(kept);
//     std::cout << "[I0_update_lucky] Updated I0_dm_runtime using top " << p
//               << "th percentile (" << kept << "/" << K << " frames).\n";
// }

// try deal better with edges!!!! 
void N0_update() {
    std::lock_guard<std::mutex> lock(telemetry_mutex);

    Eigen::VectorXd mean_dm;
    if (!temporal_mean(rtc_config.telem.img_dm, mean_dm)) {
        std::cerr << "[N0_update] Error: cannot form temporal mean of img_dm.\n";
        return;
    }

    const Eigen::Index P = mean_dm.size();
    if (P == 0) {
        std::cerr << "[N0_update] Error: img_dm mean has zero length.\n";
        return;
    }
    if (rtc_config.filters.inner_pupil_filt_dm.size() != P) {
        std::cerr << "[N0_update] Error: inner_pupil_filt_dm size mismatch ("
                  << rtc_config.filters.inner_pupil_filt_dm.size() << " vs " << P << ").\n";
        return;
    }

    // Inner-pupil mean on the temporally averaged DM vector
    constexpr double THRESH = 0.7;
    const Eigen::ArrayXd mask =
        (rtc_config.filters.inner_pupil_filt_dm.array() > THRESH).cast<double>();
    const double n_inside = mask.sum();
    if (n_inside <= 0.0) {
        std::cerr << "[N0_update] Error: no valid pixels inside inner pupil mask.\n";
        return;
    }
    const double inner_mean = (mean_dm.array() * mask).sum() / n_inside;

    // Set N0_dm_runtime to a constant vector of size P filled with inner_mean
    rtc_config.N0_dm_runtime = Eigen::VectorXd::Constant(P, inner_mean);

    std::cout << "[N0_update] Updated N0_dm_runtime (len=" << P
              << ", value=" << inner_mean << ").\n";
}

// void N0_update() {
//     std::lock_guard<std::mutex> lock(telemetry_mutex);

//     Eigen::VectorXd mean_dm;    // <rtc_config.telem.img_dm>, already per-frame normalized
//     if (!temporal_mean(rtc_config.telem.img_dm, mean_dm)) {
//         std::cerr << "[N0_update] Error: cannot form temporal mean of img_dm.\n";
//         return;
//     }

//     const Eigen::Index P = mean_dm.size();
//     if (P == 0) {
//         std::cerr << "[N0_update] Error: img_dm mean has zero length.\n";
//         return;
//     }
//     if (rtc_config.filters.inner_pupil_filt_dm.size() != P) {
//         std::cerr << "[N0_update] Error: inner_pupil_filt_dm size mismatch ("
//                   << rtc_config.filters.inner_pupil_filt_dm.size() << " vs " << P << ").\n";
//         return;
//     }

//     // Inner-pupil mean on the temporally averaged DM vector
//     constexpr double THRESH = 0.7;
//     const Eigen::ArrayXd mask = (rtc_config.filters.inner_pupil_filt_dm.array() > THRESH).cast<double>();
//     const double n_inside = mask.sum();
//     if (n_inside <= 0.0) {
//         std::cerr << "[N0_update] Error: no valid pixels inside inner pupil mask.\n";
//         return;
//     }
//     const double inner_mean = (mean_dm.array() * mask).sum() / n_inside;

//     // Replace exterior pixels with inner_mean
//     mean_dm = (mask * mean_dm.array() + (1.0 - mask) * inner_mean).matrix();

//     // Runtime projected clear-pupil reference (same normalization space)
//     rtc_config.N0_dm_runtime = mean_dm; // in the rtc we already normalize the img_dm by the sum(subframe) so no need to repeat here ! we just update with the average telem.img_dm image  

//     std::cout << "[N0_update] Updated N0_dm_runtime (len=" << P << ").\n";
// }

void dark_update() {
    pauseRTC();  // stop the loop while we replace the dark

    Eigen::VectorXd mean_img;  // average of RAW frames (ADU)
    bool ok = false;

    {
        // Limit the mutex lifetime to this scope block
        std::lock_guard<std::mutex> lock(telemetry_mutex);

        // Compute temporal mean of the RAW telemetry buffer
        if (!temporal_mean(rtc_config.telem.img, mean_img)) {
            std::cerr << "[dark_update] Error: telemetry img buffer is empty or inconsistent.\n";
        } else if (mean_img.size() <= 0) {
            std::cerr << "[dark_update] Error: mean_img has zero length.\n";
        } else {
            // Optional sanity: check against the current image_length if you track it
            // if (mean_img.size() != rtc_config.image_length) { ...warn... }

            rtc_config.reduction.dark = mean_img;  // store per-pixel dark in ADU
            ok = true;
        }
    } // mutex released here

    if (ok) {
        std::cerr << "[dark_update] reduction.dark updated from telem.img (P="
                  << mean_img.size() << ")\n";
    }

    resumeRTC();  // always resume, even if it failed
}
// void dark_update() {
//    
//     std::lock_guard<std::mutex> lock(telemetry_mutex);

//     Eigen::VectorXd mean_img;  // average of RAW frames (ADU)
//     if (!temporal_mean(rtc_config.telem.img, mean_img)) {
//         std::cerr << "[dark_update] Error: telemetry img buffer is empty or inconsistent.\n";
//         return;
//     }

//     rtc_config.reduction.dark = mean_img;  // store per-pixel dark in ADU
//     std::cerr << "[dark_update] reduction.dark updated from telem.img (P=" 
//               << mean_img.size() << ")\n";
//     
// }

// void N0_update() {

//     // !!! only updates rtc_config.N0_dm_runtime !!! 
//     // does not update rtc_config.reference_pupils members 

//     // added 
//     //Eigen::VectorXd mean_full;  // <rtc_config.telem.img> used for sub frame normalization 


//     std::lock_guard<std::mutex> lock(telemetry_mutex);

//     if (rtc_config.telem.img_dm.empty()) {
//         std::cerr << "[N0_update] Error: telemetry img_dm buffer is empty." << std::endl;
//         return;
//     }

//     const auto& telemetry_frames = rtc_config.telem.img_dm;
//     Eigen::Index P = telemetry_frames[0].size();  // size of each img_dm vector

//     if (rtc_config.filters.inner_pupil_filt_dm.size() != P) {
//         std::cerr << "[N0_update] Error: inner_pupil_filt_dm size mismatch." << std::endl;
//         return;
//     }

//     // spatial average all img_dm frames
//     Eigen::VectorXd avg_dm = Eigen::VectorXd::Zero(P);
//     size_t count = 0;
//     for (const auto& frame : telemetry_frames) {
//         if (frame.size() != P) {
//             std::cerr << "[N0_update] Warning: img_dm frame size mismatch, skipping." << std::endl;
//             continue;
//         }
//         avg_dm += frame;
//         ++count;
//     }
//     if (count == 0) {
//         std::cerr << "[N0_update] Error: no valid img_dm frames found." << std::endl;
//         return;
//     }
//     avg_dm /= static_cast<double>(count);

//     // Compute the mean inside the good pupil
//     double sum_inside = 0.0;
//     int count_inside = 0;
//     for (Eigen::Index i = 0; i < P; ++i) {
//         if (rtc_config.filters.inner_pupil_filt_dm(i) > 0.7) {
//             sum_inside += avg_dm(i);
//             ++count_inside;
//         }
//     }
//     if (count_inside == 0) {
//         std::cerr << "[N0_update] Error: no valid pixels inside inner pupil mask." << std::endl;
//         return;
//     }
//     double inner_mean = sum_inside / static_cast<double>(count_inside);

//     // Replace exterior pixels to the mean of the interior - this is to avoid bad edge and division by zero issues
//     for (Eigen::Index i = 0; i < P; ++i) {
//         if (rtc_config.filters.inner_pupil_filt_dm(i) <= 0.7) {
//             avg_dm(i) = inner_mean;
//         }
//     }

//     // // normalize by the average subframe sum 
//     //     if (!temporal_mean(rtc_config.telem.img, mean_full)) {
//     //     std::cerr << "[N0_update] Error: cannot form temporal mean of img (needed for normalization).\n";
//     //     return;
//     // }

//     // // Scalar normalization S = sum(<img>)
//     // const double S = mean_full.sum();
//     // if (!std::isfinite(S) || S <= 0.0) {
//     //     std::cerr << "[N0_update] Error: invalid normalization scalar S=" << S
//     //               << " (need finite and > 0). Aborting update.\n";
//     //     return;
//     // }


//     // Save into runtime only
//     rtc_config.N0_dm_runtime = avg_dm // / S ;

//     std::cout << "[N0_update] Successfully updated N0_dm_runtime based on img_dm telemetry." << std::endl;
// }


// Function to update the capacity of the telemetry image ring buffer.
json set_telem_capacity(json args) {
    // Expect one argument: [new_capacity]
    if (!args.is_array() || args.size() != 1) {
        return json{{"error", "Expected an array with one element: [new_capacity]"}};
    }
    try {
        size_t newCapacity = args.at(0).get<size_t>();

        // Lock the telemetry mutex during the update.
        std::lock_guard<std::mutex> lock(telemetry_mutex);
        rtc_config.telem.setCapacity(newCapacity);
        
        std::cout << "Telemetry ring buffers capacity updated to " << newCapacity << std::endl;
        return json{{"status", "Telemetry capacity updated to " + std::to_string(newCapacity)}};
    }
    catch (const std::exception &ex) {
        std::cerr << "Error updating telemetry capacity: " << ex.what() << std::endl;
        return json{{"error", ex.what()}};
    }
}

void save_telemetry(){
    rtc_config.state.take_telemetry = 1;
}

json set_telem_save_path(json args) {
    if (!args.is_array() || args.size() != 1) {
        return json{{"error", "Expected an array with one element: [\"/desired/save/path/\"]"}};
    }
    try {
        std::string new_path = args.at(0).get<std::string>();
        if (new_path.empty()) {
            return json{{"error", "Provided path is empty."}};
        }

        // Ensure path ends with a slash '/'
        if (new_path.back() != '/') {
            new_path += '/';
        }

        telem_save_path = new_path;
        // {
        //     std::lock_guard<std::mutex> lock(telemetry_mutex);
        //     telem_save_path = new_path;
        // }

        // create the directory now if it doesn't exist
        try {
            std::filesystem::create_directories(new_path);
        } catch (const std::exception& ex) {
            return json{{"error", "Failed to create directory: " + std::string(ex.what())}};
        }

        std::cout << "[set_telem_save_path] Updated telemetry save path to: " << new_path << std::endl;
        return json{{"status", "Telemetry save path updated", "new_path", new_path}};
    }
    catch (const std::exception& ex) {
        return json{{"error", ex.what()}};
    }
}


void pauseRTC() {
    pause_rtc.store(true);
    std::cout << "RTC paused." << std::endl;
}

void resumeRTC() {
    {
        std::lock_guard<std::mutex> lock(rtc_pause_mutex);
        pause_rtc.store(false);
    }
    rtc_pause_cv.notify_all();
    std::cout << "RTC resumed." << std::endl;
}


json set_telem_save_format(json args) {
    // Check that the argument is an array with exactly one element.
    if (!args.is_array() || args.size() != 1) {
        return json{{"error", "Expected an array with one element: [\"json\" or \"fits\"]"}};
    }
    try {
        // Get the format string from the argument.
        std::string format = args.at(0).get<std::string>();
        // Convert to lowercase to allow case-insensitive matching.
        std::transform(format.begin(), format.end(), format.begin(), ::tolower);
        // Check that format is either "json" or "fits".
        if (format != "json" && format != "fits") {
            return json{{"error", "Invalid format. Please use \"json\" or \"fits\"."}};
        }
        // Update the global telemFormat variable.
        telemFormat = format;
        std::cout << "Telemetry output format updated to: " << telemFormat << std::endl;
        return json{{"status", "Telemetry format updated successfully"}};
    } catch (const std::exception& ex) {
        return json{{"error", ex.what()}};
    }
}

// comment out pid dependancies to try first compile, re-incorporate later 20-8-25
// This Commander function accepts one JSON parameter: an array of three items:
// [ gain_type, indices, value ]
// e.g. in interactive shell : > update_pid_param ["LO","kp", "all", 0.0] or  ["HO","ki","all",0.2] or update_pid_param ["HO","ki","1,3,5",0.2] to update gains of modes 1,3,5 
// json update_pid_param(json args) {
//     // Expect an array with exactly 4 elements:
//     // [controller, parameter_type, indices, value]
//     if (!args.is_array() || args.size() != 4) {
//         return json{{"error", "Expected an array with four elements: [controller, parameter_type, indices, value]"}};
//     }
//     try {
//         std::string controller_str = args.at(0).get<std::string>();
//         std::string parameter_type = args.at(1).get<std::string>();
//         std::string indices_str = args.at(2).get<std::string>();
//         double value = args.at(3).get<double>();

//         // Lock the PID mutex to protect concurrent access.
//         std::lock_guard<std::mutex> lock(ctrl_mutex);

//         // Select the appropriate PID controller from rtc_config.
//         PIDController* pid_ctrl = nullptr;
//         if (controller_str == "LO") {
//             pid_ctrl = &rtc_config.ctrl_LO;
//         } else if (controller_str == "HO") {
//             pid_ctrl = &rtc_config.ctrl_HO;
//         } else {
//             return json{{"error", "Invalid controller type: " + controller_str + ". Must be \"LO\" or \"HO\"."}};
//         }
        
//         // Choose the target parameter vector.
//         Eigen::VectorXd* target_vector = nullptr;
//         if (parameter_type == "kp") {
//             target_vector = &pid_ctrl->kp;
//         } else if (parameter_type == "ki") {
//             target_vector = &pid_ctrl->ki;
//         } else if (parameter_type == "kd") {
//             target_vector = &pid_ctrl->kd;
//         } else if (parameter_type == "set_point") {
//             target_vector = &pid_ctrl->set_point;
//         } else if (parameter_type == "lower_limits") {
//             target_vector = &pid_ctrl->lower_limits;
//         } else if (parameter_type == "upper_limits") {
//             target_vector = &pid_ctrl->upper_limits;
//         } else {
//             return json{{"error", "Invalid parameter type: " + parameter_type +
//                                      ". Use one of \"kp\", \"ki\", \"kd\", \"set_point\", \"lower_limits\", or \"upper_limits\"."}};
//         }
        
//         int n = target_vector->size();
//         if (indices_str == "all") {
//             target_vector->setConstant(value);
//             std::cout << "Updated all elements of " << parameter_type << " in controller " << controller_str 
//                       << " to " << value << std::endl;
//         } else {
//             // Use your split_indices helper function to split indices_str by commas.
//             std::vector<int> idxs = split_indices(indices_str);
//             if (idxs.empty()) {
//                 return json{{"error", "No valid indices provided"}};
//             }
//             for (int idx : idxs) {
//                 if (idx < 0 || idx >= n) {
//                     return json{{"error", "Index " + std::to_string(idx) + " out of range for " + parameter_type 
//                                         + " vector of size " + std::to_string(n)}};
//                 }
//                 (*target_vector)(idx) = value;
//             }
//             std::cout << "Updated indices (" << indices_str << ") of " << parameter_type << " in controller " 
//                       << controller_str << " with value " << value << std::endl;
//         }
//         return json{{"status", "success"}};
//     }
//     catch (const std::exception &ex) {
//         return json{{"error", ex.what()}};
//     }
// }
// Rename of your former update_pid_param; now works with unique_ptr<Controller>.
// Expected args: [ "LO" | "HO", param_name, "all" | "i,j,k", value ]
// param_name supported: kp, ki, kd, set_point (also accepts "setpoint"), lower_limits, upper_limits
nlohmann::json update_ctrl_param(nlohmann::json args) {
    using json = nlohmann::json;

    // Validate args shape
    if (!args.is_array() || args.size() != 4) {
        return json{{"error", "Expected [controller, parameter_type, indices, value]"}};
    }

    try {
        std::string controller_str = args.at(0).get<std::string>();
        std::string parameter_type = args.at(1).get<std::string>();
        std::string indices_str    = args.at(2).get<std::string>();
        double value               = args.at(3).get<double>();

        // normalize
        auto to_lower = [](std::string s){
            for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        controller_str = to_lower(controller_str);
        parameter_type = to_lower(parameter_type);
        if (parameter_type == "setpoint") parameter_type = "set_point";

        // choose controller
        std::unique_ptr<Controller>* which = nullptr;
        if (controller_str == "lo")      which = &rtc_config.ctrl_LO;
        else if (controller_str == "ho") which = &rtc_config.ctrl_HO;
        else return json{{"error", "Invalid controller: use \"LO\" or \"HO\""}};

        if (!(*which)) return json{{"error", "Controller is not initialized"}};

        // allow only vector params
        static const std::unordered_set<std::string> allowed = {
            "kp","ki","kd","set_point","lower_limits","upper_limits","alpha",
            // uncomment if you want to allow state edits via commander:
            // "integrals","prev_errors","output"
        };
        if (!allowed.count(parameter_type)) {
            return json{{"error", "Invalid parameter: " + parameter_type}};
        }

        std::lock_guard<std::mutex> lock(ctrl_mutex);

        // fetch vector, edit, write back
        Controller::Param p = (*which)->get_parameter(parameter_type);
        if (!std::holds_alternative<Eigen::VectorXd>(p)) {
            return json{{"error", "Parameter is not a vector: " + parameter_type}};
        }
        Eigen::VectorXd v = std::get<Eigen::VectorXd>(p);
        const int n = static_cast<int>(v.size());

        if (indices_str == "all") {
            v.setConstant(value);
        } else {
            // use your existing helper
            std::vector<int> idxs = split_indices(indices_str);
            if (idxs.empty()) {
                return json{{"error", "No valid indices provided"}};
            }
            for (int idx : idxs) {
                if (idx < 0 || idx >= n) {
                    return json{{"error", "Index " + std::to_string(idx) +
                                             " out of range for " + parameter_type +
                                             " (size " + std::to_string(n) + ")"}};
                }
                v(idx) = value;
            }
        }

        (*which)->set_parameter(parameter_type, v);

        return json{
            {"status", "success"},
            {"controller", controller_str},
            {"parameter", parameter_type},
            {"size", n},
            {"indices", indices_str},
            {"value", value}
        };
    } catch (const std::exception& ex) {
        return json{{"error", ex.what()}};
    }
}

// // relative incrememnt of gains 
// json dg(json args) {
//     // Expect an array with exactly 3 elements: [controller, parameter_type, increment]
//     if (!args.is_array() || args.size() != 3) {
//         return json{{"error", "Expected an array with three elements: [controller, parameter_type, increment]"}};
//     }
//     try {
//         std::string controller_str = args.at(0).get<std::string>();
//         std::string parameter_type = args.at(1).get<std::string>();
//         double increment_value = args.at(2).get<double>();

//         // Lock the PID mutex to protect concurrent access
//         std::lock_guard<std::mutex> lock(ctrl_mutex);

//         // Select appropriate PID controller
//         PIDController* pid_ctrl = nullptr;
//         if (controller_str == "LO") {
//             pid_ctrl = &rtc_config.ctrl_LO;
//         } else if (controller_str == "HO") {
//             pid_ctrl = &rtc_config.ctrl_HO;
//         } else {
//             return json{{"error", "Invalid controller type: " + controller_str + ". Must be \"LO\" or \"HO\"."}};
//         }

//         // Choose the target parameter vector
//         Eigen::VectorXd* target_vector = nullptr;
//         if (parameter_type == "kp") {
//             target_vector = &pid_ctrl->kp;
//         } else if (parameter_type == "ki") {
//             target_vector = &pid_ctrl->ki;
//         } else if (parameter_type == "kd") {
//             target_vector = &pid_ctrl->kd;
//         } else {
//             return json{{"error", "Invalid parameter type: " + parameter_type +
//                                      ". Must be \"kp\", \"ki\", or \"kd\"."}};
//         }

//         // Increment all elements
//         *target_vector += Eigen::VectorXd::Constant(target_vector->size(), increment_value);

//         std::cout << "[increment_pid_param] Incremented " << parameter_type
//                   << " in controller " << controller_str
//                   << " by " << increment_value << std::endl;

//         return json{{"status", "Increment applied successfully"}};
//     }
//     catch (const std::exception& ex) {
//         return json{{"error", ex.what()}};
//     }
// }
nlohmann::json dg(nlohmann::json args) {
    using json = nlohmann::json;

    // Accept 3 or 4 arguments
    if (!args.is_array() || (args.size() != 3 && args.size() != 4)) {
        return json{{"error", "Expected [controller, parameter_type, increment] or [controller, parameter_type, indices, increment]"}};
    }

    try {
        // Parse required fields
        std::string controller_str = args.at(0).get<std::string>();
        std::string parameter_type = args.at(1).get<std::string>();

        // Normalize
        auto to_lower = [](std::string s){
            for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        controller_str = to_lower(controller_str);
        parameter_type = to_lower(parameter_type);

        // Controller pointer (polymorphic)
        std::unique_ptr<Controller>* which = nullptr;
        if (controller_str == "lo")      which = &rtc_config.ctrl_LO;
        else if (controller_str == "ho") which = &rtc_config.ctrl_HO;
        else return json{{"error", "Invalid controller: use \"LO\" or \"HO\""}};

        if (!(*which)) return json{{"error", "Controller is not initialized"}};

        // Only gains for dg
        static const std::unordered_set<std::string> allowed = {"kp","ki","kd"};
        if (!allowed.count(parameter_type)) {
            return json{{"error", "Invalid parameter for dg: " + parameter_type + " (use kp|ki|kd)"}};
        }

        // Parse indices/increment depending on arity
        std::string indices_str = "all";
        double delta = 0.0;
        if (args.size() == 3) {
            // ["LO","kp", 0.02]
            delta = args.at(2).get<double>();
        } else {
            // ["LO","kp","1,3,5", 0.02]  or  ["LO","kp","all", 0.02]
            indices_str = args.at(2).get<std::string>();
            delta       = args.at(3).get<double>();
        }

        std::lock_guard<std::mutex> lock(ctrl_mutex);

        // Fetch, modify, write back
        Controller::Param p = (*which)->get_parameter(parameter_type);
        if (!std::holds_alternative<Eigen::VectorXd>(p)) {
            return json{{"error", "Parameter is not a vector: " + parameter_type}};
        }
        Eigen::VectorXd v = std::get<Eigen::VectorXd>(p);
        const int n = static_cast<int>(v.size());

        if (indices_str == "all") {
            v.array() += delta;
        } else {
            std::vector<int> idxs = split_indices(indices_str); // your helper: "1,3,5" -> {1,3,5}
            if (idxs.empty()) return json{{"error", "No valid indices provided"}};
            for (int idx : idxs) {
                if (idx < 0 || idx >= n) {
                    return json{{"error", "Index " + std::to_string(idx) +
                                         " out of range for " + parameter_type +
                                         " (size " + std::to_string(n) + ")"}};
                }
                v(idx) += delta;
            }
        }

        (*which)->set_parameter(parameter_type, v);

        return json{
            {"status", "success"},
            {"controller", controller_str},
            {"parameter", parameter_type},
            {"indices", indices_str},
            {"delta", delta},
            {"size", n}
        };
    } catch (const std::exception& ex) {
        return json{{"error", ex.what()}};
    }
}



// json print_pid_attribute(json args) {
//     // Expect exactly two elements: [controller, attribute]
//     if (!args.is_array() || args.size() != 2) {
//         return json{{"error", "Expected an array with two elements: [controller, attribute]"}};
//     }
    
//     try {
//         std::string controller_str = args.at(0).get<std::string>();
//         std::string attribute = args.at(1).get<std::string>();
        
//         // Select the appropriate PID controller from rtc_config.
//         PIDController* ctrl = nullptr;
//         if (controller_str == "LO") {
//             ctrl = &rtc_config.ctrl_LO;
//         } else if (controller_str == "HO") {
//             ctrl = &rtc_config.ctrl_HO;
//         } else {
//             return json{{"error", "Invalid controller specified. Must be \"LO\" or \"HO\""}};
//         }

//         // lock it while we read 
//         std::lock_guard<std::mutex> lock(ctrl_mutex);


//         // Helper lambda to convert an Eigen::VectorXd to a std::vector<double>.
//         auto eigen_to_vector = [](const Eigen::VectorXd &v) -> std::vector<double> {
//             return std::vector<double>(v.data(), v.data() + v.size());
//         };
        

//         if (attribute == "kp") {
//             return json{{"kp", eigen_to_vector(ctrl->kp)}};
//         } else if (attribute == "ki") {
//             return json{{"ki", eigen_to_vector(ctrl->ki)}};
//         } else if (attribute == "kd") {
//             return json{{"kd", eigen_to_vector(ctrl->kd)}};
//         } else if (attribute == "lower_limits") {
//             return json{{"lower_limits", eigen_to_vector(ctrl->lower_limits)}};
//         } else if (attribute == "upper_limits") {
//             return json{{"upper_limits", eigen_to_vector(ctrl->upper_limits)}};
//         } else if (attribute == "set_point") {
//             return json{{"set_point", eigen_to_vector(ctrl->set_point)}};
//         } else if (attribute == "output") {
//             return json{{"output", eigen_to_vector(ctrl->output)}};
//         } else if (attribute == "integrals") {
//             return json{{"integrals", eigen_to_vector(ctrl->integrals)}};
//         } else if (attribute == "prev_errors") {
//             return json{{"prev_errors", eigen_to_vector(ctrl->prev_errors)}};
//         } else {
//             return json{{"error", "Unknown attribute: " + attribute}};
//         }
//     } catch (const std::exception &ex) {
//         return json{{"error", ex.what()}};
//     }
// }

Eigen::VectorXd dm140_to_144(const Eigen::VectorXd& input140) {
    if (input140.size() != 140) {
        throw std::runtime_error("Input vector must be of size 140");
    }

    Eigen::VectorXd output144(144);
    int j = 0;
    for (int i = 0; i < 144; ++i) {
        // Skip corners: top-left (0), top-right (11), bottom-left (132), bottom-right (143)
        if (i == 0 || i == 11 || i == 132 || i == 143) {
            output144(i) = 0.0;
        } else {
            output144(i) = input140(j++);
        }
    }
    return output144;
}

Eigen::VectorXd dm144_to_140(const Eigen::Ref<const Eigen::VectorXd>& v144)
{
    if (v144.size() != 144)
        throw std::runtime_error("dm144_to_140: input size must be 144");

    Eigen::VectorXd out(140);
    int j = 0;
    for (int i = 0; i < 144; ++i) {
        // Skip the four corner indices in row-major order:
        // (0,0)->0, (0,11)->11, (11,0)->132, (11,11)->143
        if (i == 0 || i == 11 || i == 132 || i == 143) continue;
        out[j++] = v144[i];
    }
    // j should be 140 here
    return out;
}

void build_interaction_matrix(double poke_amp = 0.05, int num_iterations = 10, double sleep_seconds = 0.01, const std::string& signal_space = "dm", const std::string& output_filename = "") {
    std::cout << "[IM] Starting interaction matrix capture with 140 modes..." << std::endl;

    // Take new dark and bias
    //new_dark_and_bias();

    // Get gain and fps
    float gain = get_float_cam_param("gain");
    float fps = get_float_cam_param("fps");
    double darkscale = fps / gain;

    int nx = subarray.md->size[0];
    int ny = (subarray.md->naxis > 1) ? subarray.md->size[1] : 1;
    int totalPixels = nx * ny;
    int N_modes = 140;

    Eigen::MatrixXd IM(totalPixels, N_modes);

    for (int k = 0; k < N_modes; ++k) {
        std::cout << "[IM] Poking actuator " << k << "/" << N_modes << std::endl;

        Eigen::VectorXd cmd = Eigen::VectorXd::Zero(N_modes);
        cmd(k) = poke_amp / 2.0;

        Eigen::VectorXd padded_cmd(144);
        padded_cmd.setZero();
        int idx = 0;
        for (int i = 0; i < 144; ++i) {
            if (!(i == 0 || i == 11 || i == 132 || i == 143)) {
                padded_cmd(i) = cmd(idx++);
            }
        }

        std::vector<Eigen::VectorXd> Iplus_list, Iminus_list;

        for (int iter = 0; iter < num_iterations; ++iter) {
            // Positive poke
            updateDMSharedMemory(dm_rtc, padded_cmd);
            ImageStreamIO_sempost(&dm_rtc0, 1);
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_seconds));

            Eigen::VectorXd Iplus = Eigen::VectorXd::Zero(totalPixels);
            for (int i = 0; i < 200; ++i) {
                catch_up_with_sem(&subarray, 1);
                ImageStreamIO_semwait(&subarray, 1);
                //uint16_t *raw = subarray.array.UI16;
                //Eigen::Map<const Eigen::Array<uint16_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
                int32_t* raw = subarray.array.SI32;
                Eigen::Map<const Eigen::Array<int32_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
                Eigen::VectorXd img = rawArr.cast<double>();
                img -= (darkscale * rtc_config.reduction.dark + rtc_config.reduction.bias);
                Iplus += img;
            }
            Iplus /= 200.0;
            Iplus_list.push_back(Iplus);

            // Negative poke
            updateDMSharedMemory(dm_rtc, -padded_cmd);
            ImageStreamIO_sempost(&dm_rtc0, 1);
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep_seconds));

            Eigen::VectorXd Iminus = Eigen::VectorXd::Zero(totalPixels);
            for (int i = 0; i < 200; ++i) {
                catch_up_with_sem(&subarray, 1);
                ImageStreamIO_semwait(&subarray, 1);
                //uint16_t *raw = subarray.array.UI16;
                //Eigen::Map<const Eigen::Array<uint16_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
                int32_t* raw = subarray.array.SI32;
                Eigen::Map<const Eigen::Array<int32_t, Eigen::Dynamic, 1>> rawArr(raw, totalPixels);
                Eigen::VectorXd img = rawArr.cast<double>();
                img -= (darkscale * rtc_config.reduction.dark + rtc_config.reduction.bias);
                Iminus += img;
            }
            Iminus /= 200.0;
            Iminus_list.push_back(Iminus);
        }

        Eigen::VectorXd Iplus_mean = Eigen::VectorXd::Zero(totalPixels);
        Eigen::VectorXd Iminus_mean = Eigen::VectorXd::Zero(totalPixels);
        for (int i = 0; i < num_iterations; ++i) {
            Iplus_mean += Iplus_list[i];
            Iminus_mean += Iminus_list[i];
        }
        Iplus_mean /= static_cast<double>(num_iterations);
        Iminus_mean /= static_cast<double>(num_iterations);

        Eigen::VectorXd errsig = (Iplus_mean - Iminus_mean) * darkscale / poke_amp;
        IM.col(k) = errsig;
    }

    std::string filename = output_filename.empty() ? "/home/asg/Music/IM_test.fits" : output_filename;
    write_eigen_matrix_to_fits(IM, filename, "IM");
    std::cout << "[IM] Saved interaction matrix to " << filename << std::endl;
}




json reload_config(json args) {
    // Expect two arguments: [new_filename, new_phase_mask]. filename MUST be .toml from the Baldr calibration pipeline
    if (!args.is_array() || args.size() != 2) {
        return json{{"error", "Expected an array with two elements: [new_filename, new_phase_mask]"}};
    }
    try {
        // Extract new configuration file name and phase mask.
        std::string newFilename = args.at(0).get<std::string>();
        std::string newPhaseMask = args.at(1).get<std::string>();

        // Pause the RTC loop.
        pauseRTC();
        std::cout << "RTC paused for configuration reload." << std::endl;

        // Parse the new TOML configuration file.
        config = toml::parse_file(newFilename);
        std::cout << "Loaded new configuration file: " << newFilename << std::endl;
        g_active_config_filename = newFilename; // put in global variable

        // Update the phase mask globally.
        phasemask = newPhaseMask;

        // Construct the beam key based on the current beam number.
        std::string beamKey = "beam" + std::to_string(beam_id);

        // Update rtc_config using the new configuration.
        rtc_config = readBDRConfig(config, beamKey, phasemask);
        std::cout << "rtc_config updated using beam " << beamKey << " and phase mask " << phasemask << std::endl;

        // Reinitialize PID controllers using the new controller configuration.
        {
            //std::lock_guard<std::mutex> lock1(ctrl_mutex);
            //std::lock_guard<std::mutex> lock2(telemetry_mutex);
            std::scoped_lock lock(ctrl_mutex, telemetry_mutex); //C++17
            // ctrl_LO = PIDController(rtc_config.ctrl_LO_config);
            // ctrl_HO = PIDController(rtc_config.ctrl_HO_config);
            // std::cout << "PID controllers reinitialized: "
            //           << "ctrl_LO.kp.size()=" << ctrl_LO.kp.size() << ", "
            //           << "ctrl_HO.kp.size()=" << ctrl_HO.kp.size() << std::endl;
            rtc_config.initDerivedParameters();

        }

        
        //rtc_config.telem = bdr_telem();  // Reset telemetry buffers.

        // Resume the RTC loop.
        resumeRTC();
        std::cout << "RTC resumed after configuration reload." << std::endl;

        return json{{"status", "New configuration loaded successfully"}};
    }
    catch (const std::exception &ex) {
        std::cerr << "Error reloading configuration: " << ex.what() << std::endl;
        // Make sure to resume if necessary, or signal an error state.
        resumeRTC();
        return json{{"error", ex.what()}};
    }
}

/// NEed to test: poking in control space so we can build IM in closed loop 
// void closed_loop_modal_id(const std::string& space = "dm", int num_frames = 20, double delta = 0.1, const std::string& output_filename = "/home/asg/Music/modal_response.fits") {
//     std::cout << "[Modal ID] Starting closed-loop identification for modes..." << std::endl;

//     // Determine which telemetry buffer to use
//     const bool use_dm_space = (space == "dm");
//     const auto& telemetry_buffer = use_dm_space ? rtc_config.telem.img_dm : rtc_config.telem.imgs;

//     if (telemetry_buffer.empty()) {
//         std::cerr << "[Modal ID] Error: telemetry buffer is empty." << std::endl;
//         return;
//     }

//     const int N_modes_LO = rtc_config.ctrl_LO.set_point.size();
//     const int N_modes_HO = rtc_config.ctrl_HO.set_point.size();
//     const int N_total_modes = N_modes_LO + N_modes_HO;
//     const int frame_size = telemetry_buffer.front().size();

//     std::vector<Eigen::MatrixXd> response_data;

//     // Lock to modify setpoints
//     std::lock_guard<std::mutex> lock(ctrl_mutex);

//     for (int k = 0; k < N_total_modes; ++k) {
//         std::cout << "[Modal ID] Probing mode " << k << "/" << N_total_modes << std::endl;

//         PIDController& ctrl = (k < N_modes_LO) ? rtc_config.ctrl_LO : rtc_config.ctrl_HO;
//         int local_k = (k < N_modes_LO) ? k : k - N_modes_LO;

//         // Helper lambda to collect telemetry after a setpoint change
//         auto collect = [&](double sign) -> Eigen::MatrixXd {
//             ctrl.set_point(local_k) = sign * delta;
//             std::this_thread::sleep_for(std::chrono::milliseconds(500));

//             std::vector<Eigen::VectorXd> collected;
//             for (int i = 0; i < num_frames; ++i) {
//                 if (telemetry_buffer.size() < 1) {
//                     std::cerr << "[Modal ID] Warning: no telemetry available during read." << std::endl;
//                     continue;
//                 }
//                 collected.push_back(telemetry_buffer.back());
//                 std::this_thread::sleep_for(std::chrono::milliseconds(10));
//             }

//             Eigen::MatrixXd result(frame_size, collected.size());
//             for (int i = 0; i < collected.size(); ++i)
//                 result.col(i) = collected[i];
//             return result;
//         };

//         // Positive response
//         Eigen::MatrixXd pos_resp = collect(+1);
//         response_data.push_back(pos_resp);

//         // Negative response
//         Eigen::MatrixXd neg_resp = collect(-1);
//         response_data.push_back(neg_resp);

//         // Reset setpoint
//         ctrl.set_point(local_k) = 0.0;
//     }

//     // Write to FITS file (one HDU per entry)
//     fitsfile* fptr;
//     int status = 0;
//     fits_create_file(&fptr, output_filename.c_str(), &status);
//     if (status) {
//         fits_report_error(stderr, status);
//         return;
//     }

//     for (int i = 0; i < response_data.size(); ++i) {
//         const auto& mat = response_data[i];
//         long naxes[2] = {mat.rows(), mat.cols()};
//         long fpixel[2] = {1, 1};

//         if (i == 0) {
//             fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status);
//         } else {
//             fits_create_img(fptr, DOUBLE_IMG, 2, naxes, &status);
//         }

//         fits_write_pix(fptr, TDOUBLE, fpixel, naxes[0]*naxes[1], (void*)mat.data(), &status);
//     }

//     fits_close_file(fptr, &status);
//     std::cout << "[Modal ID] Saved closed-loop response data to " << output_filename << std::endl;
// }

//////////////////

// 
namespace { // anonymous namespace fpr probe_methods saving
// #include <algorithm>
// #include <cctype>
// #include <string>
// #include <string_view>
// #include <vector>
// #include <nlohmann/json.hpp>

inline bool iequals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    return std::equal(a.begin(), a.end(), b.begin(),
        [](char x, char y){
            return std::tolower(static_cast<unsigned char>(x)) ==
                   std::tolower(static_cast<unsigned char>(y));
        });
}

// Accepts "all", a single name, or ["name1","name2",...]
inline std::vector<std::string> parse_field_list(const nlohmann::json& jfield) {
    if (jfield.is_string()) {
        std::string s = jfield.get<std::string>();
        if (iequals(s, "all")) return list_all_numeric_telem_fields();
        return {std::move(s)};
    }
    if (jfield.is_array()) {
        std::vector<std::string> out;
        out.reserve(jfield.size());
        for (const auto& x : jfield) out.push_back(x.get<std::string>());
        return out;
    }
    throw std::runtime_error(
        "`field` must be a string ('all' or a field name) or a list of strings.");
}
} // end anonymous namespace


namespace dm {

// fits files that populate this struc are created here :
// asgard-alignment/playground/build_DM_basis_configs.py
// also should be copied to dcs/util but will require some modules from asgard-alginment python package to run!

// Immutable snapshot of all loaded bases.
struct DMBasisSet {
    long nx = 0;   // expected NX for 3D cubes; 0 if unknown
    long ny = 0;   // expected NY for 3D cubes; 0 if unknown
    // basis name (lowercase) -> modes (each Eigen::VectorXd is flattened row-major of length nx*ny)
    std::unordered_map<std::string, std::shared_ptr<const std::vector<Eigen::VectorXd>>> bases;
};

// Inline variables are OK in C++17
inline std::shared_ptr<const DMBasisSet> g_dm_bases = std::make_shared<DMBasisSet>();


// Helpers
inline void publish(std::shared_ptr<const DMBasisSet> nv) {
    std::atomic_store_explicit(&g_dm_bases, std::move(nv), std::memory_order_release);
}
inline std::shared_ptr<const DMBasisSet> snapshot() {
    return std::atomic_load_explicit(&g_dm_bases, std::memory_order_acquire);
}

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}



// Internal: read a single BASIS extension currently selected in fptr.
// Accepts 3D (NX,NY,NMODES) or 2D (LEN,NMODES). Returns vector of flattened modes and (nx,ny).
inline bool read_current_basis_ext(fitsfile* fptr,
                                   std::vector<Eigen::VectorXd>& out_modes,
                                   long& out_nx, long& out_ny,
                                   std::string& warn_msg)
{
    int status = 0;
    out_modes.clear(); out_nx = out_ny = 0; warn_msg.clear();

    int naxis = 0;
    if (fits_get_img_dim(fptr, &naxis, &status)) return false;
    long naxes[3] = {0,0,0};
    if (fits_get_img_size(fptr, naxis, naxes, &status)) return false;

    // Read entire image as double
    long nelem = 1; for (int i=0;i<naxis;++i) nelem *= naxes[i];
    std::vector<double> buf(static_cast<size_t>(nelem));
    long fpixel = 1; double nulval = 0.0; int anynul = 0;
    if (fits_read_img(fptr, TDOUBLE, fpixel, nelem, &nulval, buf.data(), &anynul, &status))
        return false;

    // 2D fallback: (LEN, NMODES)
    if (naxis == 2) {
        const long LEN = naxes[0], NMODES = naxes[1];
        out_nx = LEN; out_ny = 1;
        out_modes.reserve(static_cast<size_t>(NMODES));
        for (long k=0;k<NMODES;++k) {
            const long base = k*LEN;
            Eigen::VectorXd v(LEN);
            for (long i=0;i<LEN;++i) v[i] = buf[base+i];
            out_modes.emplace_back(std::move(v));
        }
        warn_msg = "2D (LEN,NMODES) treated as pre-flattened.";
        return true;
    }
    if (naxis != 3) { warn_msg = "Unsupported NAXIS (expected 2 or 3)."; return false; }

    // Identify which axis is "modes"
    int mode_ax = -1;
    {   // Prefer header key NMODES if present
        LONGLONG nmods_ll = -1;
        if (fits_read_key_lnglng(fptr, "NMODES", &nmods_ll, nullptr, &status) == 0) {
            long nmods = static_cast<long>(nmods_ll);
            for (int a=0;a<3;++a) if (naxes[a] == nmods) { mode_ax = a; break; }
        }
        status = 0; // ignore missing key
    }
    if (mode_ax < 0) {
        // Heuristic: take the largest dim as modes (typical: 140 vs 12×12)
        mode_ax = int(std::distance(naxes, std::max_element(naxes, naxes+3)));
    }

    // Remaining two axes are spatial; the lower axis index varies fastest -> X
    int sp[2], si = 0; for (int a=0;a<3;++a) if (a != mode_ax) sp[si++] = a;
    std::sort(sp, sp+2);
    const int ax_x = sp[0], ax_y = sp[1];

    const long NX = naxes[ax_x], NY = naxes[ax_y], NMODES = naxes[mode_ax];
    out_nx = NX; out_ny = NY;
    out_modes.reserve(static_cast<size_t>(NMODES));

    // FITS linear index: i = i1 + N1*(i2 + N2*i3), with i1 = axis1 (0-based), etc.
    const long stride1 = 1;
    const long stride2 = naxes[0];
    const long stride3 = naxes[0]*naxes[1];

    for (long k=0;k<NMODES;++k) {
        Eigen::VectorXd v(NX*NY);
        for (long y=0;y<NY;++y) {
            for (long x=0;x<NX;++x) {
                long idx3[3] = {0,0,0};
                idx3[ax_x]    = x;
                idx3[ax_y]    = y;
                idx3[mode_ax] = k;
                const long lin = idx3[0]*stride1 + idx3[1]*stride2 + idx3[2]*stride3;
                v[y*NX + x] = buf[lin];
            }
        }
        out_modes.emplace_back(std::move(v));
    }
    if (mode_ax != 2) warn_msg = "Mode axis not NAXIS3; auto-detected.";
    return true;
}

// Load all BASIS:* image extensions from a FITS file into a new snapshot.
// On any error, prints a warning and returns false (leaves previous snapshot unchanged).
inline bool load_from_fits_file(const std::string& fits_path)
{
    int status = 0;
    fitsfile* fptr = nullptr;
    auto close_guard = [&]{ if (fptr) fits_close_file(fptr, &status); };

    if (fits_open_file(&fptr, fits_path.c_str(), READONLY, &status)) {
        char msg[FLEN_STATUS]; fits_get_errstatus(status, msg);
        std::cerr << "[DMBasis] WARN: cannot open FITS '" << fits_path << "': " << msg << "\n";
        close_guard();
        return false;
    }

    int hdunum = 0;
    if (fits_get_num_hdus(fptr, &hdunum, &status)) {
        std::cerr << "[DMBasis] WARN: FITS get_num_hdus failed.\n"; close_guard(); return false;
    }

    auto nv = std::make_shared<DMBasisSet>();
    long common_nx = 0, common_ny = 0;
    int n_loaded = 0;

    for (int i = 2; i <= hdunum; ++i) { // start at 2 (skip primary)
        if (fits_movabs_hdu(fptr, i, nullptr, &status)) { status = 0; continue; }

        char extname[FLEN_VALUE] = {0};
        if (fits_read_key(fptr, TSTRING, "EXTNAME", extname, nullptr, &status)) {
            status = 0; continue;
        }
        std::string en(extname);
        if (en.size() < 7 || en.rfind("BASIS:", 0) != 0) continue; // not starting with BASIS:

        std::string key = lower(en.substr(6));  // after "BASIS:"
        std::vector<Eigen::VectorXd> modes;
        long nx = 0, ny = 0;
        std::string warn;
        if (!read_current_basis_ext(fptr, modes, nx, ny, warn)) {
            std::cerr << "[DMBasis] WARN: failed to read extension '" << en << "'\n";
            continue;
        }
        if (!warn.empty()) std::cerr << "[DMBasis] NOTE: " << warn << " (EXTNAME=" << en << ")\n";

        if (modes.empty()) {
            std::cerr << "[DMBasis] WARN: extension '" << en << "' has zero modes; skipping.\n";
            continue;
        }

        if (common_nx == 0 && ny > 0) { // first successful ext with 3D or 2D
            common_nx = nx;
            common_ny = ny;
        } else {
            // Enforce consistent lengths; allow 2D LENx1 to match NX*NY if equal
            const long lenA = (common_ny > 1) ? (common_nx * common_ny) : common_nx;
            const long lenB = (ny > 1) ? (nx * ny) : nx;
            if (lenB != lenA) {
                std::cerr << "[DMBasis] WARN: extension '" << en
                          << "' length " << lenB << " != " << lenA
                          << "; skipping to keep consistent geometry.\n";
                continue;
            }
        }

        // Store (share) without copying modes again
        nv->bases.emplace(std::move(key),
                          std::make_shared<const std::vector<Eigen::VectorXd>>(std::move(modes)));
        ++n_loaded;
    }

    if (n_loaded == 0) {
        std::cerr << "[DMBasis] WARN: no BASIS:* extensions found in '" << fits_path << "'.\n";
        close_guard();
        return false;
    }

    // If common_ny == 1 we only know flat length; encode as (LEN,1)
    nv->nx = common_nx;
    nv->ny = common_ny;

    publish(nv);
    close_guard();
    std::cerr << "[DMBasis] Loaded " << n_loaded << " bases from '" << fits_path
              << "' (nx=" << nv->nx << ", ny=" << nv->ny << ").\n";
    return true;
}

// Convenience: try load once if cache is empty; returns whether cache is non-empty afterwards.
inline bool ensure_loaded_from(const std::string& fits_path)
{
    auto snap = snapshot();
    if (!snap || snap->bases.empty()) {
        (void)load_from_fits_file(fits_path); // ignore boolean; we only warn
        snap = snapshot();
        return (snap && !snap->bases.empty());
    }
    return true;
}

// to decouple the rtc.cpp (which uses dm namespace for signal injection stuff)
// from the file path and keep the default path private to baldr.cpp.
bool ensure_loaded_default() {
    return ensure_loaded_from(kDMBaseFITS);
} // new

// Lookup (no copy). Returns nullptr if basis missing or cache empty.
inline std::shared_ptr<const std::vector<Eigen::VectorXd>>
get_basis(const std::string& basis_key_lower)
{
    auto snap = snapshot();
    if (!snap) return nullptr;
    auto it = snap->bases.find(lower(basis_key_lower));
    if (it == snap->bases.end()) return nullptr;
    return it->second;
}

} // namespace dm


static std::string upper(std::string s){
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::toupper(c); });
    return s;
}

inline std::vector<std::string> list_loaded_bases() {
    std::vector<std::string> out;
    auto snap = dm::snapshot();
    if (!snap) return out;
    out.reserve(snap->bases.size());
    for (auto& kv : snap->bases) out.push_back(kv.first);
    // Print nicely
    std::cout << "[DMBasis] Loaded bases (" << out.size() << "): ";
    for (size_t i = 0; i < out.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << out[i];
    }
    std::cout << '\n';
    return out;
}

json ol_set_mode(const json& args)
{
    //ol_set_mode — set/add an open-loop offset in HO or LO using a basis mode; usage: ["HO|LO","<basis>",<index>,<amplitude>,(optional)<accumulate_bool>], e.g. ["HO","zernike",5,0.2] or ["LO","hadamard",12,-0.05,true]
    try {
        // ---- Parse args ----
        std::string branch, basis;
        long long   idx_ll = -1;
        double      amplitude = 0.0;
        bool        accumulate = false;

        if (args.is_array()) {
            if (args.size() < 4 || args.size() > 5)
                return json{{"ok", false}, {"error", "usage: [\"HO|LO\", basis, index, amplitude, (optional) accumulate]"}};
            branch = args.at(0).get<std::string>();
            basis  = args.at(1).get<std::string>();
            if (args.at(2).is_number_integer() || args.at(2).is_number_unsigned())
                idx_ll = args.at(2).get<long long>();
            else if (args.at(2).is_string())
                idx_ll = std::stoll(args.at(2).get<std::string>());
            else
                return json{{"ok", false}, {"error", "index must be integer"}};
            if (args.at(3).is_number())
                amplitude = args.at(3).get<double>();
            else if (args.at(3).is_string())
                amplitude = std::stod(args.at(3).get<std::string>());
            else
                return json{{"ok", false}, {"error", "amplitude must be numeric"}};
            if (args.size() == 5) accumulate = args.at(4).get<bool>();
        } else if (args.is_object()) {
            if (!args.contains("branch") || !args.contains("basis") ||
                !args.contains("index")  || !args.contains("amplitude"))
                return json{{"ok", false}, {"error", "usage: {branch,basis,index,amplitude[,accumulate]}"}};
            branch = args["branch"].get<std::string>();
            basis  = args["basis"].get<std::string>();
            if (args["index"].is_number_integer() || args["index"].is_number_unsigned())
                idx_ll = args["index"].get<long long>();
            else if (args["index"].is_string())
                idx_ll = std::stoll(args["index"].get<std::string>());
            else
                return json{{"ok", false}, {"error", "index must be integer"}};
            if (args["amplitude"].is_number())
                amplitude = args["amplitude"].get<double>();
            else if (args["amplitude"].is_string())
                amplitude = std::stod(args["amplitude"].get<std::string>());
            else
                return json{{"ok", false}, {"error", "amplitude must be numeric"}};
            if (args.contains("accumulate")) accumulate = args["accumulate"].get<bool>();
        } else if (args.is_string()) {
            std::istringstream iss(args.get<std::string>());
            if (!(iss >> branch >> basis >> idx_ll >> amplitude))
                return json{{"ok", false}, {"error", "usage: \"HO <basis> <index> <amp> [accumulate]\""}};
            if (!iss.eof()) iss >> std::boolalpha >> accumulate;
        } else {
            return json{{"ok", false}, {"error", "args must be array, object, or string"}};
        }

        const std::string BR = upper(branch);
        if (BR != "HO" && BR != "LO")
            return json{{"ok", false}, {"error", "branch must be \"HO\" or \"LO\""}};
        if (idx_ll < 0)
            return json{{"ok", false}, {"error", "index must be >= 0"}};

        // ---- Ensure OL initialized ----
        if (!ol_init_if_needed())
            return json{{"ok", false}, {"error", "offsets not initialized (DM length unknown)"}};
        auto cur = std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
        if (!cur) return json{{"ok", false}, {"error", "internal null snapshot"}};
        const int dm_len = static_cast<int>(cur->lo.size());

        // ---- Ensure bases loaded (once) ----
        dm::ensure_loaded_from(kDMBaseFITS); // will WARN but not throw on failure
        auto snap = dm::snapshot();
        if (!snap || snap->bases.empty())
            return json{{"ok", false}, {"error", "no DM bases loaded"}, {"file", kDMBaseFITS}};

        // ---- Lookup basis ----
        auto modes_ptr = dm::get_basis(basis); // lowercase lookup inside
        if (!modes_ptr)
            return json{{"ok", false}, {"error", "basis not found"}, {"basis", basis}};

        const auto& modes = *modes_ptr;
        if (idx_ll >= static_cast<long long>(modes.size()))
            return json{{"ok", false}, {"error", "index out of range"}, {"n_modes", (long)modes.size()}};

        const Eigen::VectorXd& mode = modes[static_cast<size_t>(idx_ll)];
        if (mode.size() != dm_len)
            return json{{"ok", false}, {"error", "basis mode length != DM length"},
                        {"basis_len", (long)mode.size()}, {"dm_len", dm_len}};

        // ---- Apply ----
        Eigen::VectorXd v = amplitude * mode;
        if (BR == "HO") {
            if (accumulate) set_openloop_offset_HO(cur->ho + v);
            else            set_openloop_offset_HO(v);
        } else {
            if (accumulate) set_openloop_offset_LO(cur->lo + v);
            else            set_openloop_offset_LO(v);
        }

        return json{{"ok", true},
                    {"branch", BR},
                    {"basis", basis},
                    {"index", idx_ll},
                    {"amplitude", amplitude},
                    {"accumulate", accumulate},
                    {"dm_len", dm_len}, {"n_modes", (long)modes.size()}};
    }
    catch (const std::exception& e) {
        return json{{"ok", false}, {"error", std::string("parse/apply: ") + e.what()}};
    }
}


// Helper: classify the field
static inline int telem_field_kind(const std::string& f) {
    // 0 = vector<Eigen::VectorXd>, 1 = scalar<double>, 2 = scalar<int>, -1 = unknown
    if (f=="img" || f=="img_dm" || f=="signal" || f=="e_LO" || f=="u_LO" ||
        f=="e_HO" || f=="u_HO" || f=="c_LO" || f=="c_HO") return 0;
    if (f=="timestamp" || f=="rmse_est" || f=="snr") return 1;
    if (f=="LO_servo_mode" || f=="HO_servo_mode")   return 2;
    return -1;
}

// capture N samples of baldr telemetry field , by default we change telemetry capacity to match N 
std::vector<std::vector<double>>
capture_telem_list(const std::string& field, size_t N, long long dead_ms, bool change_capacity = true)
{
    if (N == 0) return {};

    const int kind = telem_field_kind(field);
    if (kind < 0)
        throw std::invalid_argument("capture_telem_list: unknown field \"" + field + "\"");

    size_t old_cap = 0;

    // Optionally change capacity just for this call
    if (change_capacity) {
        std::lock_guard<std::mutex> lk(telemetry_mutex);
        old_cap = rtc_config.telem.signal.capacity(); // all ring buffers share capacity
        rtc_config.telem.setCapacity(N);
    }

    // If dead_ms < 0, compute a wait that roughly fills N samples; otherwise use given dead_ms
    if (dead_ms < 0) {
        const double fps = (rtc_config.fps > 1e-9 ? rtc_config.fps : 1000.0);
        const double ms = 1000.0 * (double)N / std::max(1e-9, fps) * 1.2; // 120% margin
        dead_ms = (long long)std::llround(std::min(10000.0, std::max(0.0, ms)));
    }
    if (dead_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(dead_ms));

    // Snapshot ONLY the requested field (oldest -> newest)
    std::vector<std::vector<double>> out;
    {
        std::lock_guard<std::mutex> lk(telemetry_mutex);

        auto dump_vecbuf = [&](const boost::circular_buffer<Eigen::VectorXd>& buf) {
            const size_t have  = buf.size();
            const size_t take  = std::min(N, have);
            const size_t start = have >= take ? (have - take) : 0;
            out.reserve(take);
            for (size_t i = 0; i < take; ++i) {
                const Eigen::VectorXd& v = buf[start + i];
                out.emplace_back(v.data(), v.data() + v.size());
            }
        };
        auto dump_dblbuf = [&](const boost::circular_buffer<double>& buf) {
            const size_t have  = buf.size();
            const size_t take  = std::min(N, have);
            const size_t start = have >= take ? (have - take) : 0;
            out.reserve(take);
            for (size_t i = 0; i < take; ++i)
                out.push_back(std::vector<double>{ buf[start + i] });
        };
        auto dump_intbuf = [&](const boost::circular_buffer<int>& buf) {
            const size_t have  = buf.size();
            const size_t take  = std::min(N, have);
            const size_t start = have >= take ? (have - take) : 0;
            out.reserve(take);
            for (size_t i = 0; i < take; ++i)
                out.push_back(std::vector<double>{ static_cast<double>(buf[start + i]) });
        };

        switch (kind) {
            case 0: // vector fields
                if      (field=="img")      dump_vecbuf(rtc_config.telem.img);
                else if (field=="img_dm")   dump_vecbuf(rtc_config.telem.img_dm);
                else if (field=="signal")   dump_vecbuf(rtc_config.telem.signal);
                else if (field=="e_LO")     dump_vecbuf(rtc_config.telem.e_LO);
                else if (field=="u_LO")     dump_vecbuf(rtc_config.telem.u_LO);
                else if (field=="e_HO")     dump_vecbuf(rtc_config.telem.e_HO);
                else if (field=="u_HO")     dump_vecbuf(rtc_config.telem.u_HO);
                else if (field=="c_LO")     dump_vecbuf(rtc_config.telem.c_LO);
                else if (field=="c_HO")     dump_vecbuf(rtc_config.telem.c_HO);
                else if (field=="c_inj")    dump_vecbuf(rtc_config.telem.c_inj);  // <-- add this
                break;
            case 1: // scalar<double>
                if      (field=="timestamp") dump_dblbuf(rtc_config.telem.timestamp);
                else if (field=="rmse_est")  dump_dblbuf(rtc_config.telem.rmse_est);
                else if (field=="snr")       dump_dblbuf(rtc_config.telem.snr);
                break;
            case 2: // scalar<int>
                if      (field=="LO_servo_mode") dump_intbuf(rtc_config.telem.LO_servo_mode);
                else if (field=="HO_servo_mode") dump_intbuf(rtc_config.telem.HO_servo_mode);
                break;
        }
    }

    // Restore capacity if we changed it here
    if (change_capacity) {
        std::lock_guard<std::mutex> lk(telemetry_mutex);
        rtc_config.telem.setCapacity(old_cap);
    }

    return out;
}

nlohmann::json capture_telem_list_json(const nlohmann::json& args) {
    using json = nlohmann::json;
    try {
        if (!args.is_array() || args.size() < 2 || args.size() > 4)
            return json{{"ok", false}, {"error", "usage: [field, N, dead_ms=0, change_capacity=true]"}};

        const std::string field = args.at(0).get<std::string>();
        const size_t N          = args.at(1).get<size_t>();
        long long dead_ms       = (args.size() >= 3) ? args.at(2).get<long long>() : 0;
        bool change_capacity    = (args.size() >= 4) ? args.at(3).get<bool>() : true;

        auto data = capture_telem_list(field, N, dead_ms, change_capacity);
        return json{{"ok", true}, {"field", field}, {"N", N}, {"data", std::move(data)}};
    } catch (const std::exception& e) {
        return json{{"ok", false}, {"error", std::string("capture_telem_list_json: ") + e.what()}};
    }
}


/// upgrading the above (but keeping for backwards compatibility) to include list of filds
// Capture N samples for multiple fields ("all" | one name | list of names)
std::unordered_map<std::string, std::vector<std::vector<double>>>
capture_telem_multi(const std::vector<std::string>& fields, size_t N, long long dead_ms, bool change_capacity = true)
{
    std::unordered_map<std::string, std::vector<std::vector<double>>> out;
    if (N == 0 || fields.empty()) return out;

    size_t old_cap = 0;

    // Optionally change capacity once for all fields
    if (change_capacity) {
        std::lock_guard<std::mutex> lk(telemetry_mutex);
        old_cap = rtc_config.telem.signal.capacity(); // all ring buffers share capacity
        rtc_config.telem.setCapacity(N);
    }

    // Compute a default wait to roughly fill N samples if needed
    if (dead_ms < 0) {
        const double fps = (rtc_config.fps > 1e-9 ? rtc_config.fps : 1000.0);
        const double ms  = 1000.0 * (double)N / std::max(1e-9, fps) * 1.2; // 120% margin
        dead_ms = (long long)std::llround(std::min(10000.0, std::max(0.0, ms)));
    }
    if (dead_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(dead_ms));

    // Snapshot all requested fields under one mutex
    {
        std::lock_guard<std::mutex> lk(telemetry_mutex);

        auto dump_vecbuf = [&](const boost::circular_buffer<Eigen::VectorXd>& buf,
                               std::vector<std::vector<double>>& dst) {
            const size_t have  = buf.size();
            const size_t take  = std::min(N, have);
            const size_t start = have >= take ? (have - take) : 0;
            dst.clear(); dst.reserve(take);
            for (size_t i = 0; i < take; ++i) {
                const Eigen::VectorXd& v = buf[start + i];
                dst.emplace_back(v.data(), v.data() + v.size());
            }
        };
        auto dump_dblbuf = [&](const boost::circular_buffer<double>& buf,
                               std::vector<std::vector<double>>& dst) {
            const size_t have  = buf.size();
            const size_t take  = std::min(N, have);
            const size_t start = have >= take ? (have - take) : 0;
            dst.clear(); dst.reserve(take);
            for (size_t i = 0; i < take; ++i)
                dst.push_back(std::vector<double>{ buf[start + i] });
        };
        auto dump_intbuf = [&](const boost::circular_buffer<int>& buf,
                               std::vector<std::vector<double>>& dst) {
            const size_t have  = buf.size();
            const size_t take  = std::min(N, have);
            const size_t start = have >= take ? (have - take) : 0;
            dst.clear(); dst.reserve(take);
            for (size_t i = 0; i < take; ++i)
                dst.push_back(std::vector<double>{ static_cast<double>(buf[start + i]) });
        };

        for (const auto& field : fields) {
            auto& dst = out[field]; // creates entry
            const int kind = telem_field_kind(field);
            if (kind < 0) {
                // Unknown field: erase entry and continue (or throw if you prefer)
                out.erase(field);
                continue;
            }
            switch (kind) {
                case 0: // vector fields
                    if      (field=="img")        dump_vecbuf(rtc_config.telem.img, dst);
                    else if (field=="img_dm")     dump_vecbuf(rtc_config.telem.img_dm, dst);
                    else if (field=="signal")     dump_vecbuf(rtc_config.telem.signal, dst);
                    else if (field=="e_LO")       dump_vecbuf(rtc_config.telem.e_LO, dst);
                    else if (field=="u_LO")       dump_vecbuf(rtc_config.telem.u_LO, dst);
                    else if (field=="e_HO")       dump_vecbuf(rtc_config.telem.e_HO, dst);
                    else if (field=="u_HO")       dump_vecbuf(rtc_config.telem.u_HO, dst);
                    else if (field=="c_LO")       dump_vecbuf(rtc_config.telem.c_LO, dst);
                    else if (field=="c_HO")       dump_vecbuf(rtc_config.telem.c_HO, dst);
                    else if (field=="c_inj")      dump_vecbuf(rtc_config.telem.c_inj, dst); // new
                    break;
                case 1: // scalar<double>
                    if      (field=="timestamp")  dump_dblbuf(rtc_config.telem.timestamp, dst);
                    else if (field=="rmse_est")   dump_dblbuf(rtc_config.telem.rmse_est,  dst);
                    else if (field=="snr")        dump_dblbuf(rtc_config.telem.snr,       dst);
                    break;
                case 2: // scalar<int>
                    if      (field=="LO_servo_mode") dump_intbuf(rtc_config.telem.LO_servo_mode, dst);
                    else if (field=="HO_servo_mode") dump_intbuf(rtc_config.telem.HO_servo_mode, dst);
                    break;
            }
        }
    }

    if (change_capacity) {
        std::lock_guard<std::mutex> lk(telemetry_mutex);
        rtc_config.telem.setCapacity(old_cap);
    }

    return out;
}


nlohmann::json capture_telem_multi_json(const nlohmann::json& args) {
    using json = nlohmann::json;
    try {
        // Usage: [fields_spec, N, dead_ms=0, change_capacity=true]
        if (!args.is_array() || args.size() < 2 || args.size() > 4)
            return json{{"ok", false}, {"error", "usage: [fields, N, dead_ms=0, change_capacity=true]"}};

        // fields_spec can be "all" | "<name>" | ["a","b",...]
        std::vector<std::string> fields = parse_field_list(args.at(0));

        const size_t N          = args.at(1).get<size_t>();
        long long dead_ms       = (args.size() >= 3) ? args.at(2).get<long long>() : 0;
        bool change_capacity    = (args.size() >= 4) ? args.at(3).get<bool>() : true;

        auto data = capture_telem_multi(fields, N, dead_ms, change_capacity);
        return json{{"ok", true}, {"fields", fields}, {"N", N}, {"data", std::move(data)}};
    } catch (const std::exception& e) {
        return json{{"ok", false}, {"error", std::string("capture_telem_multi_json: ") + e.what()}};
    }
}


// new multi field 
nlohmann::json probe_interaction_data(nlohmann::json args)
{
    using json = nlohmann::json;

    bool all_requested = false;   // NEW: remember if caller asked for "all"

    // -------- parse --------
    std::vector<std::pair<long long,double>> aberr;
    std::string basis, state;
    std::vector<std::string> fields;    // supports "all" | name | ["a","b",...]
    std::string field_single;           // legacy single-field JSON path
    size_t N = 0;
    long long dead_ms = -1;

    try {
        if (args.is_array()) {
            // [ [[idx,amp],...], basis, state, field|'all'|[...], N, deadtime? ]
            if (args.size() < 5)
                return json{{"ok",false},{"error","usage: [[[idx,amp],...], basis, state, field|'all'|[...], N, deadtime?]"}};

            basis  = args.at(1).get<std::string>();
            state  = args.at(2).get<std::string>();

            const auto& farg = args.at(3);
            if (farg.is_string()) {
                std::string s = farg.get<std::string>();
                if (iequals(s, "all")) all_requested = true;              // <- NEW
            }
            fields = parse_field_list(farg);                               // "all" | name | ["a","b",...]
            // fields = parse_field_list(args.at(3));           // NEW: accepts "all" | name | ["a","b",...]
            N      = args.at(4).get<size_t>();
            if (args.size() >= 6) dead_ms = args.at(5).get<long long>();

            const auto& arr = args.at(0);
            if (!arr.is_array()) return json{{"ok",false},{"error","aberrations must be array"}};
            for (const auto& t : arr) {
                if (!t.is_array() || t.size()!=2) return json{{"ok",false},{"error","each aberration must be [idx,amp]"}};
                long long idx = t.at(0).get<long long>();
                double amp    = t.at(1).get<double>();
                aberr.emplace_back(idx, amp);
            }
        } else if (args.is_object()) {
            basis  = args.at("basis").get<std::string>();
            state  = args.at("state").get<std::string>();
            N      = args.at("N").get<size_t>();
            // if (auto it = args.find("field"); it != args.end()) fields = parse_field_list(*it);
            // else                                                fields = list_all_numeric_telem_fields();
            // if (args.contains("deadtime_ms")) dead_ms = args.at("deadtime_ms").get<long long>();
            if (auto it = args.find("field"); it != args.end()) { // have to deal with case of "all" input so FIELD headers isnt too long and just writes "all"
                const auto& farg = *it;
                if (farg.is_string()) {
                    std::string s = farg.get<std::string>();
                    if (iequals(s, "all")) all_requested = true;          // <- NEW
                }
                fields = parse_field_list(farg);
            } else {
                fields = list_all_numeric_telem_fields();
                all_requested = true;                                      // <- NEW (missing field ⇒ treat as "all")
            }

            if (args.contains("deadtime_ms")) dead_ms = args.at("deadtime_ms").get<long long>();

            const auto& arr = args.at("aberrations");
            if (!arr.is_array()) return json{{"ok",false},{"error","aberrations must be array"}};
            for (const auto& t : arr) {
                if (!t.is_array() || t.size()!=2) return json{{"ok",false},{"error","each aberration must be [idx,amp]"}};
                long long idx = t.at(0).get<long long>();
                double amp    = t.at(1).get<double>();
                aberr.emplace_back(idx, amp);
            }
        } else {
            return json{{"ok",false},{"error","bad args"}};
        }
    } catch (const std::exception& e) {
        return json{{"ok",false},{"error",std::string("parse: ")+e.what()}};
    }

    const bool single_field_mode = (fields.size() == 1);
    if (single_field_mode) field_single = fields[0];

    // Compute outfile safely (object may provide it; arrays won’t)
    std::string outfile = "probe_result.fits";
    if (args.is_object() && args.contains("outfile")) {
        try { outfile = args.at("outfile").get<std::string>(); } catch (...) {}
    }

    // -------- prepare global telemetry capacity ONCE --------
    size_t old_cap = 0;
    {
        std::lock_guard<std::mutex> lk(telemetry_mutex);
        old_cap = rtc_config.telem.signal.capacity();
        rtc_config.telem.setCapacity(N);
    }
    auto restore_capacity = [&](){
        std::lock_guard<std::mutex> lk(telemetry_mutex);
        if (rtc_config.telem.signal.capacity() != old_cap)
            rtc_config.telem.setCapacity(old_cap);
    };

    // -------- run probes --------
    json out;
    out["ok"]          = true;
    out["basis"]       = basis;
    out["state"]       = state;
    if (single_field_mode) out["field"] = field_single; else out["fields"] = fields;
    out["N"]           = static_cast<long long>(N);
    out["deadtime_ms"] = dead_ms;
    out["n_probes"]    = static_cast<long long>(aberr.size());
    out["probes"]      = json::array();

    try {
        for (const auto& [idx_ll, amp] : aberr) {
            // Apply mode
            if (state == "OL" || state == "ol" || state == "Ol") {
                json japply = ol_set_mode(json::array({"HO", basis, idx_ll, amp}));
                if (!japply.value("ok", false)) {
                    restore_capacity();
                    return json{{"ok",false},{"error","ol_set_mode failed"},{"detail",japply}};
                }
            } else if (state == "CL" || state == "cl" || state == "Cl") {
                auto modes_ptr = dm::get_basis(basis);
                if (!modes_ptr) {
                    restore_capacity();
                    return json{{"ok",false},{"error","basis not found"},{"basis",basis}};
                }
                const auto& modes = *modes_ptr;
                if (idx_ll < 0 || static_cast<size_t>(idx_ll) >= modes.size()) {
                    restore_capacity();
                    return json{{"ok",false},{"error","index out of range"},{"idx",idx_ll},{"n_modes",(long)modes.size()}};
                }
                Eigen::VectorXd mode144 = modes[static_cast<size_t>(idx_ll)];
                mode144 *= amp;
                Eigen::VectorXd mode140 = dm144_to_140(mode144);
                json jset = ctrl_set(json::array({"HO","set_point", eigen_vector_to_json(mode140)}));
                if (!jset.value("ok", false)) {
                    restore_capacity();
                    return json{{"ok",false},{"error","ctrl_set set_point failed"},{"detail",jset}};
                }
            } else {
                restore_capacity();
                return json{{"ok",false},{"error","Invalid state (use 'OL' or 'CL')"},{"state",state}};
            }

            // dead time
            if (dead_ms < 0) {
                const double fps = (rtc_config.fps > 1e-9 ? rtc_config.fps : 1000.0);
                const double ms  = 1000.0 * (double)N / std::max(1e-9, fps) * 1.2; // 120% margin
                std::this_thread::sleep_for(std::chrono::milliseconds((long long)std::llround(std::min(5000.0, std::max(0.0, ms)))));
            } else if (dead_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(dead_ms));
            }

            // Capture ONLY for legacy single-field mode
            if (single_field_mode) {
                std::vector<std::vector<double>> frames =
                    capture_telem_list(field_single, N, /*dead_ms*/dead_ms, /*change_capacity*/false);

                json probe = json::object();
                probe["index"]     = idx_ll;
                probe["amplitude"] = amp;
                probe["samples"]   = json::array();
                for (const auto& v : frames) probe["samples"].push_back(v);

                out["probes"].push_back(std::move(probe));
            } else {
                json probe = json::object();
                probe["index"]     = idx_ll;
                probe["amplitude"] = amp;
                out["probes"].push_back(std::move(probe));
            }
        }
    } catch (const std::exception& e) {
        restore_capacity();
        return json{{"ok",false},{"error",std::string("probe loop: ")+e.what()}};
    }

    // -------- finalize --------
    restore_capacity();

    if (!single_field_mode) {
        // Multi-field / "all": write one FITS with one IMAGE HDU per field
        try {
            // Build primary-HDU headers mirroring legacy JSON
            std::vector<std::pair<std::string, std::string>> hdr_strs = {
                {"BASIS", basis},
                {"STATE", state}
            };

            if (all_requested) {
                hdr_strs.emplace_back("FIELDS", "all");                    // <- NEW
            } else if (fields.size() == 1) {
                hdr_strs.emplace_back("FIELD", fields[0]);
            } else {
                // Comma-separated list only when user gave an explicit subset
                std::string fields_csv;
                for (size_t i = 0; i < fields.size(); ++i) {
                    if (i) fields_csv += ",";
                    fields_csv += fields[i];
                }
                hdr_strs.emplace_back("FIELDS", fields_csv);
            }
            // if (fields.size() == 1) {
            //     hdr_strs.emplace_back("FIELD", fields[0]);
            // } else {
            //     // Comma-separated list of fields
            //     std::string fields_csv;
            //     for (size_t i = 0; i < fields.size(); ++i) {
            //         if (i) fields_csv += ",";
            //         fields_csv += fields[i];
            //     }
            //     hdr_strs.emplace_back("FIELDS", fields_csv);
            // }
            std::vector<std::pair<std::string, long>> hdr_longs = {
                {"N",        static_cast<long>(N)},
                {"DEADTIME", static_cast<long>(dead_ms)},
                {"NPROBES",  static_cast<long>(aberr.size())}
            };
            std::vector<std::pair<std::string, double>> hdr_dbls; // none for now

            auto extractor = [&](std::string_view name, Eigen::MatrixXd& M, std::string& why) {
                return telem_extract_matrix_lastN(rtc_config.telem, telemetry_mutex, name, N, M, why);
            };


            // --- Camera + Boxcar headers for primary HDU ---
            try {
                const double gain = static_cast<double>(get_float_cam_param("gain raw"));
                const double fps  = static_cast<double>(get_float_cam_param("fps raw"));
                hdr_dbls.emplace_back("GAINRAW", gain);
                hdr_dbls.emplace_back("FPSRAW",  fps);
            } catch (...) {
                // ignore if camera params unavailable
                std::cerr << "[probe] camera params unavailable; skipping GAINRAW/FPSRAW\n";
            }

            try {
                long nbox = static_cast<long>(std::llround(global_boxcar.load()));
                hdr_longs.emplace_back("NBOXCAR", nbox);
                // Optional float value too:
                // hdr_dbls.emplace_back("GBOXCAR", static_cast<double>(global_boxcar.load()));
            } catch (...) {
                // ignore if unavailable
                std::cerr << "[probe] boxcar unavailable; skipping NBOXCAR\n";
            }
            const bool ok = write_fields_to_fits(
                outfile, fields, extractor, static_cast<long>(N),
                hdr_strs, hdr_longs, hdr_dbls
            );

            if (!ok) {
                return json{{"ok",false},
                            {"error","failed to write multi-field FITS (write_fields_to_fits)"},
                            {"outfile",outfile},
                            {"fields",fields},
                            {"N",(long long)N}};
            }
            out["mode"]    = "multi-field";
            out["outfile"] = outfile;
            return out;
        } catch (const std::exception& e) {
            return json{{"ok",false},{"error",std::string("save failure: ")+e.what()}};
        }
    }

    // Legacy single-field JSON return (unchanged)
    out["mode"] = "single-field";
    return out;
}
// previous single field 

// // Probe interaction data by applying a sequence of modal aberrations and capturing N samples.

// // TO DO: Add all method to get all telemetry .. need to update save_probe_result_fits

// // Args (JSON):
// //   array form: [aberrations, basis, state, field, N, deadtime_ms(optional)]
// //     - aberrations: e.g. [[5, 0.05], [12, -0.03], ...]   (idx, amplitude)
// //     - basis:       string (e.g. "zernike", "fourier_12x12", ...)
// //     - state:       "OL" or "CL"
// //     - field:       telemetry field (e.g. "signal", "img", "e_HO", ...)
// //     - N:           number of sequential samples per aberration
// //     - deadtime_ms: optional; if <0 will be computed from fps & N; else used as-is
// //   object form: {"aberrations":[...], "basis":"...", "state":"OL|CL", "field":"...",
// //                 "N":123, "deadtime_ms":0}
// //
// // Behavior:
// //   - If state=="OL": uses ol_set_mode ["HO", basis, idx, amp] to apply, then captures N samples.
// //   - If state=="CL": builds HO set_point = amp * dm144_to_140(basis[idx]) and sets it via ctrl_set.
// //   - Capacity of all telemetry buffers is set to N once, then restored at the end.
// //   - For each aberration, state is restored (OL offsets or HO set_point) after capture.
// //
// // Returns JSON:
// //   {"ok":true, "basis":..., "state":"OL|CL", "field":..., "N":..., "deadtime_ms":..., 
// //    "probes":[ {"idx":i, "amp":a, "data":[ [...], [...], ... ]}, ... ] }
// //
// // Requires helpers already in this file:
// //   - dm::get_basis(std::string) -> std::shared_ptr<const std::vector<Eigen::VectorXd>>
// //   - dm144_to_140(const Eigen::Ref<const Eigen::VectorXd>&)
// //   - eigen_vector_to_json(const Eigen::VectorXd&)
// //   - ctrl_get(json), ctrl_set(json)
// //   - ol_set_mode(json)
// //   - capture_telem_list_json(json)  // we pass [field, N, 0, false]
// //   - telemetry_mutex, rtc_config
// //   - g_ol_offsets, set_openloop_offset_LO/HO(), ol_init_if_needed()
// nlohmann::json probe_interaction_data(nlohmann::json args)
// {
//     using json = nlohmann::json;

//     // -------- parse --------
//     std::vector<std::pair<long long,double>> aberr;
//     std::string basis, state, field;
//     size_t N = 0;
//     long long dead_ms = -1;

//     try {
//         if (args.is_array()) {
//             // [ [[idx,amp],...], basis, state, field, N, deadtime? ]
//             if (args.size() < 5) return json{{"ok",false},{"error","usage: [[[idx,amp],...],basis,state,field,N,deadtime?]"}};
//             basis = args.at(1).get<std::string>();
//             state = args.at(2).get<std::string>();
//             field = args.at(3).get<std::string>();
//             N     = args.at(4).get<size_t>();
//             if (args.size() >= 6) dead_ms = args.at(5).get<long long>();

//             const auto& arr = args.at(0);
//             if (!arr.is_array()) return json{{"ok",false},{"error","aberrations must be array"}};
//             for (const auto& t : arr) {
//                 if (!t.is_array() || t.size()!=2) return json{{"ok",false},{"error","each aberration must be [idx,amp]"}};
//                 long long idx = t.at(0).get<long long>();
//                 double amp    = t.at(1).get<double>();
//                 aberr.emplace_back(idx, amp);
//             }
//         } else if (args.is_object()) {
//             basis = args.at("basis").get<std::string>();
//             state = args.at("state").get<std::string>();
//             N     = args.at("N").get<size_t>();
//             //field = args.at("field").get<std::string>();

//             std::vector<std::string> fields;
//             try {
//                 if (auto it = args.find("field"); it != args.end()) {
//                     fields = parse_field_list(*it);          // accepts "all" | "name" | ["a","b",...]
//                 } else {
//                     // Decide your default: treat missing as "all" (recommended)
//                     fields = list_all_numeric_telem_fields();
//                 }
//             } catch (const std::exception& e) {
//                 return {{"ok", false}, {"error", e.what()}};
//             }

//             if (args.contains("deadtime_ms")) dead_ms = args.at("deadtime_ms").get<long long>();
//             const auto& arr = args.at("aberrations");
//             if (!arr.is_array()) return json{{"ok",false},{"error","aberrations must be array"}};
//             for (const auto& t : arr) {
//                 long long idx = t.at(0).get<long long>();
//                 double amp    = t.at(1).get<double>();
//                 aberr.emplace_back(idx, amp);
//             }
//         } else {
//             return json{{"ok",false},{"error","bad args"}};
//         }
//     } catch (const std::exception& e) {
//         return json{{"ok",false},{"error",std::string("parse: ")+e.what()}};
//     }

//     // -------- prepare global telemetry capacity ONCE --------
//     size_t old_cap = 0;
//     {
//         std::lock_guard<std::mutex> lk(telemetry_mutex);
//         old_cap = rtc_config.telem.signal.capacity();
//         rtc_config.telem.setCapacity(N);
//     }

//     auto restore_capacity = [&](){
//         std::lock_guard<std::mutex> lk(telemetry_mutex);
//         if (rtc_config.telem.signal.capacity() != old_cap)
//             rtc_config.telem.setCapacity(old_cap);
//     };

//     // -------- run probes --------
//     json out;
//     out["ok"]          = true;
//     out["basis"]       = basis;
//     out["state"]       = state;
//     out["field"]       = field;
//     out["N"]           = static_cast<long long>(N);
//     out["deadtime_ms"] = dead_ms;
//     out["n_probes"]    = static_cast<long long>(aberr.size());
//     out["probes"]      = json::array();

//     try {
//         for (const auto& [idx_ll, amp] : aberr) {
//             // apply mode
//             if (state == "OL" || state == "ol" || state == "Ol") {
//                 json japply = ol_set_mode(json::array({"HO", basis, idx_ll, amp}));
//                 if (!japply.value("ok", false)) {
//                     restore_capacity();
//                     return json{{"ok",false},{"error","ol_set_mode failed"},{"detail",japply}};
//                 }
//             } else if (state == "CL" || state == "cl" || state == "Cl") {
//                 // lookup basis and set controller set_point
//                 auto modes_ptr = dm::get_basis(basis);
//                 if (!modes_ptr) {
//                     restore_capacity();
//                     return json{{"ok",false},{"error","basis not found"},{"basis",basis}};
//                 }
//                 const auto& modes = *modes_ptr;
//                 if (idx_ll < 0 || static_cast<size_t>(idx_ll) >= modes.size()) {
//                     restore_capacity();
//                     return json{{"ok",false},{"error","index out of range"},{"idx",idx_ll},{"n_modes",(long)modes.size()}};
//                 }

//                 Eigen::VectorXd mode144 = modes[static_cast<size_t>(idx_ll)];
//                 mode144 *= amp;
//                 Eigen::VectorXd mode140 = dm144_to_140(mode144);

//                 json jset = ctrl_set(json::array({"HO","set_point", eigen_vector_to_json(mode140)}));
//                 if (!jset.value("ok", false)) {
//                     restore_capacity();
//                     return json{{"ok",false},{"error","ctrl_set set_point failed"},{"detail",jset}};
//                 }
//             } else {
//                 restore_capacity();
//                 return json{{"ok",false},{"error","Invalid state (use 'OL' or 'CL')"},{"state",state}};
//             }

//             // dead time
//             if (dead_ms < 0) {
//                 // infer from fps and N (same logic as capture_telem_list when dead_ms<0)
//                 const double fps = (rtc_config.fps > 1e-9 ? rtc_config.fps : 1000.0);
//                 const double ms  = 1000.0 * (double)N / std::max(1e-9, fps) * 1.2;
//                 std::this_thread::sleep_for(std::chrono::milliseconds((long long)std::llround(std::min(5000.0, std::max(0.0, ms)))));
//             } else if (dead_ms > 0) {
//                 std::this_thread::sleep_for(std::chrono::milliseconds(dead_ms));
//             }

//             // capture without changing capacity again
//             std::vector<std::vector<double>> frames = capture_telem_list(field, N, /*dead_ms*/dead_ms, /*change_capacity*/false);

//             // --- IMPORTANT: put under "samples" (what the FITS writer expects) ---
//             json probe = json::object();
//             probe["index"]    = idx_ll;
//             probe["amplitude"]= amp;
//             probe["samples"]  = json::array();     // array of arrays
//             for (const auto& v : frames) probe["samples"].push_back(v);

//             out["probes"].push_back(std::move(probe));
//         }
//     }
//     catch (const std::exception& e) {
//         restore_capacity();
//         return json{{"ok",false},{"error",std::string("probe loop: ")+e.what()}};
//     }

//     // restore telemetry capacity
//     restore_capacity();
//     return out;
// }

// static nlohmann::json probe_interaction_data(nlohmann::json args) //(const nlohmann::json& args)
// {
//     using json = nlohmann::json;
//     try {
//         // ------------------- Parse inputs -------------------
//         std::vector<std::pair<long long,double>> aberr; // (idx, amp)
//         std::string basis;
//         std::string state;
//         std::string field;
//         long long   N = -1;
//         long long   dead_ms = 1; // default 1 ms, matches your note

//         auto parse_aberr_list = [&](const json& ja) -> bool {
//             if (!ja.is_array()) return false;
//             for (const auto& item : ja) {
//                 long long idx_ll = -1;
//                 double    amp    = 0.0;

//                 if (item.is_array() && item.size() >= 2) {
//                     // [idx, amp]
//                     if (item[0].is_number_integer() || item[0].is_number_unsigned())
//                         idx_ll = item[0].get<long long>();
//                     else if (item[0].is_string())
//                         idx_ll = std::stoll(item[0].get<std::string>());
//                     else return false;

//                     if (item[1].is_number_float() || item[1].is_number_integer())
//                         amp = item[1].get<double>();
//                     else if (item[1].is_string())
//                         amp = std::stod(item[1].get<std::string>());
//                     else return false;
//                 }
//                 else if (item.is_object() && item.contains("idx") && item.contains("amp")) {
//                     if (item["idx"].is_number_integer() || item["idx"].is_number_unsigned())
//                         idx_ll = item["idx"].get<long long>();
//                     else if (item["idx"].is_string())
//                         idx_ll = std::stoll(item["idx"].get<std::string>());
//                     else return false;

//                     if (item["amp"].is_number_float() || item["amp"].is_number_integer())
//                         amp = item["amp"].get<double>();
//                     else if (item["amp"].is_string())
//                         amp = std::stod(item["amp"].get<std::string>());
//                     else return false;
//                 } else {
//                     return false;
//                 }
//                 if (idx_ll < 0) return false;
//                 aberr.emplace_back(idx_ll, amp);
//             }
//             return true;
//         };

//         if (args.is_array()) {
//             if (args.size() < 5 || args.size() > 6)
//                 return json{{"ok", false}, {"error", "usage: [aberrations, basis, state, field, N, deadtime_ms?]"}};

//             if (!parse_aberr_list(args[0]))
//                 return json{{"ok", false}, {"error", "bad aberrations list"}};
//             basis = args[1].get<std::string>();
//             state = args[2].get<std::string>();
//             field = args[3].get<std::string>();
//             N     = args[4].get<long long>();
//             if (args.size() == 6) dead_ms = args[5].get<long long>();
//         }
//         else if (args.is_object()) {
//             if (!args.contains("aberrations") || !args.contains("basis") ||
//                 !args.contains("state") || !args.contains("field") || !args.contains("N"))
//                 return json{{"ok", false}, {"error", "missing required keys"}};

//             if (!parse_aberr_list(args["aberrations"]))
//                 return json{{"ok", false}, {"error", "bad aberrations list"}};
//             basis = args["basis"].get<std::string>();
//             state = args["state"].get<std::string>();
//             field = args["field"].get<std::string>();
//             N     = args["N"].get<long long>();
//             if (args.contains("deadtime_ms")) dead_ms = args["deadtime_ms"].get<long long>();
//         }
//         else {
//             return json{{"ok", false}, {"error", "args must be array or object"}};
//         }

//         if (aberr.empty()) return json{{"ok", false}, {"error", "no aberrations"}};
//         if (N <= 0)        return json{{"ok", false}, {"error", "N must be > 0"}};

//         // Normalize state and basis
//         auto to_lower = [](std::string s){ for (auto& c : s) c = static_cast<char>(std::tolower(c)); return s; };
//         const std::string basis_lc = to_lower(basis);
//         const std::string state_lc = to_lower(state);
//         const bool use_ol = (state_lc == "ol");
//         const bool use_cl = (state_lc == "cl");
//         if (!use_ol && !use_cl)
//             return json{{"ok", false}, {"error", "state must be \"OL\" or \"CL\""}};

//         // -------------- Set telemetry capacity ONCE --------------
//         size_t old_cap = 0;
//         {
//             std::lock_guard<std::mutex> lk(telemetry_mutex);
//             old_cap = rtc_config.telem.signal.capacity(); // all ring-buffers share capacity
//             if (old_cap != static_cast<size_t>(N))
//                 rtc_config.telem.setCapacity(static_cast<size_t>(N));
//         }
//         auto restore_capacity = [&]() {
//             std::lock_guard<std::mutex> lk(telemetry_mutex);
//             if (rtc_config.telem.signal.capacity() != old_cap)
//                 rtc_config.telem.setCapacity(old_cap);
//         };

//         // ----------- Snapshot current OL or CL state -----------
//         Eigen::VectorXd ho_saved, lo_saved;
//         Eigen::VectorXd sp_saved;

//         if (use_ol) {
//             // Ensure offsets exist
//             (void)ol_init_if_needed();
//             auto cur = std::atomic_load_explicit(&g_ol_offsets, std::memory_order_acquire);
//             if (!cur) {
//                 restore_capacity();
//                 return json{{"ok", false}, {"error", "open-loop offsets not initialized"}};
//             }
//             ho_saved = cur->ho;
//             lo_saved = cur->lo;
//         } else {
//             // Read HO controller set_point via ctrl_get
//             json j_sp = ctrl_get(json::array({"HO","set_point"}));
//             if (!j_sp.value("ok", false)) {
//                 restore_capacity();
//                 return json{{"ok", false}, {"error", "failed to read HO.set_point"}, {"detail", j_sp}};
//             }
//             // Expect j_sp["value"] to be an array -> Eigen
//             if (!j_sp.contains("value") || !j_sp["value"].is_array()) {
//                 restore_capacity();
//                 return json{{"ok", false}, {"error", "HO.set_point missing/invalid"}};
//             }
//             if (!json_to_eigen_vector(j_sp["value"], sp_saved)) {
//                 restore_capacity();
//                 return json{{"ok", false}, {"error", "could not parse HO.set_point as vector"}};
//             }
//         }

//         // If CL, ensure basis is loaded and indexable
//         std::shared_ptr<const std::vector<Eigen::VectorXd>> modes_ptr;
//         if (use_cl) {
//             modes_ptr = dm::get_basis(basis_lc);
//             if (!modes_ptr) {
//                 restore_capacity();
//                 return json{{"ok", false}, {"error", "basis not loaded"}, {"basis", basis_lc}};
//             }
//         }

//         json results = json::array();

//         // ---------------- Iterate aberrations ----------------
//         for (const auto& p : aberr) {
//             const long long idx_ll = p.first;
//             const double    amp    = p.second;

//             if (use_ol) {
//                 // Apply OL offset to HO branch via your commander entrypoint
//                 json japply = ol_set_mode(json::array({"HO", basis_lc, idx_ll, amp}));
//                 if (!japply.value("ok", false)) {
//                     // Restore & capacity, then abort
//                     set_openloop_offset_HO(ho_saved);
//                     set_openloop_offset_LO(lo_saved);
//                     restore_capacity();
//                     return json{{"ok", false}, {"error", "ol_set_mode failed"}, {"detail", japply}};
//                 }
//             } else {
//                 // Build HO set_point = amp * mode140 (converted from 144 via drop corners)
//                 const auto& modes = *modes_ptr;
//                 if (idx_ll < 0 || idx_ll >= static_cast<long long>(modes.size())) {
//                     restore_capacity();
//                     return json{{"ok", false},
//                                 {"error", "index out of range"},
//                                 {"idx", idx_ll}, {"n_modes", (long)modes.size()}};
//                 }
//                 const Eigen::VectorXd& mode144 = modes[static_cast<size_t>(idx_ll)];
//                 if (mode144.size() != 144) {
//                     restore_capacity();
//                     return json{{"ok", false},
//                                 {"error", "basis mode length != 144"},
//                                 {"got", mode144.size()}};
//                 }
//                 Eigen::VectorXd mode140 = dm144_to_140(mode144);
//                 mode140 *= amp;

//                 // Replace HO set_point (not accumulate)
//                 json jset = ctrl_set(json::array({"HO","set_point", eigen_vector_to_json(mode140)}));
//                 if (!jset.value("ok", false)) {
//                     restore_capacity();
//                     return json{{"ok", false}, {"error", "ctrl_set HO.set_point failed"}, {"detail", jset}};
//                 }
//             }

//             // ----- Dead time before sampling -----
//             if (dead_ms < 0) {
//                 // auto-compute from fps & N (120% margin, clamp 0..5000 ms)
//                 const double fps = (rtc_config.fps > 1e-9 ? rtc_config.fps : 1000.0);
//                 const double ms  = 1000.0 * (double)N / std::max(1e-9, fps) * 1.2;
//                 const long long auto_ms = (long long)std::llround(std::min(5000.0, std::max(0.0, ms)));
//                 if (auto_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(auto_ms));
//             } else if (dead_ms > 0) {
//                 std::this_thread::sleep_for(std::chrono::milliseconds(dead_ms));
//             }

//             // ----- Capture N sequential samples without changing capacity -----
//             json jcap = capture_telem_list_json(json::array({field, N, 0, false}));
//             if (!jcap.value("ok", false)) {
//                 // Restore states and capacity, then abort
//                 if (use_ol) {
//                     set_openloop_offset_HO(ho_saved);
//                     set_openloop_offset_LO(lo_saved);
//                 } else {
//                     (void)ctrl_set(json::array({"HO","set_point", eigen_vector_to_json(sp_saved)}));
//                 }
//                 restore_capacity();
//                 return json{{"ok", false}, {"error", "capture_telem_list_json failed"}, {"detail", jcap}};
//             }

//             results.push_back(json{
//                 {"idx",  idx_ll},
//                 {"amp",  amp},
//                 {"data", jcap["data"]}
//             });

//             // ----- Restore state after each probe -----
//             if (use_ol) {
//                 set_openloop_offset_HO(ho_saved);
//                 set_openloop_offset_LO(lo_saved);
//             } else {
//                 (void)ctrl_set(json::array({"HO","set_point", eigen_vector_to_json(sp_saved)}));
//             }
//         }

//         // ------------- Restore capacity on success -------------
//         restore_capacity();

//         return json{
//             {"ok", true},
//             {"basis", basis_lc},
//             {"state", use_ol ? "OL" : "CL"},
//             {"field", field},
//             {"N", N},
//             {"deadtime_ms", dead_ms},
//             {"probes", results}
//         };
//     }
//     catch (const std::exception& e) {
//         return nlohmann::json{{"ok", false}, {"error", std::string("probe_interaction_data: ") + e.what()}};
//     }
// }



// copied from the one implemented in rtc.cpp (resets internal PRBS/LFSR, timers, etc.)
// void reset_signal_injection_runtime() {
//     g_inj = SignalInjRuntime{};  // zero-initialize: inited=false, clears FIFO/PRBS/frame_idx/basis_vec
//     //key point is inited becomes false -> so compute_c_inj(...) does its one-time init next frame
// }
extern void reset_signal_injection_runtime();
/**
 * Configure injection for Tip/Tilt (LO branch) with sensible ID defaults.
 * 
 * @param method    "chirp", "prbs" (accepts "probs" alias), or "sine"
 * @param amplitude scalar amplitude applied to the scalar carrier (will be clamped)
 * @param apply_to  "setpoint" (closed-loop ID) or "command" (open-loop/command-space)
 * @return          true on success, false if method/apply_to are invalid
 */
bool configure_tt_injection(const std::string& method_in,
                            double amplitude,
                            const std::string& apply_to_in)
{
    // Lowercase copies for robust matching
    auto to_lower = [](std::string s){
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return s;
    };
    std::string method   = to_lower(method_in);
    std::string apply_to = to_lower(apply_to_in);

    // Normalize common typos/aliases
    if (method == "probs") method = "prbs";

    // Validate inputs (apply_to defaults to "setpoint" if unknown)
    if (method != "chirp" && method != "prbs" && method != "sine") {
        std::cerr << "[inj] invalid method: " << method_in << " (use chirp|prbs|sine)\n";
        return false;
    }
    if (apply_to != "setpoint" && apply_to != "command") {
        std::cerr << "[inj] invalid apply_to: " << apply_to_in << " (defaulting to setpoint)\n";
        apply_to = "setpoint";
    }

    // Safety clamp — keep lots of margin below your ±0.4 hard stop
    const double A = std::clamp(amplitude, 0.0, 0.30);

    // Apply config under lock; we also reset the injection runtime state so changes take effect cleanly.
    {
        std::lock_guard<std::mutex> lk(rtc_mutex);
        auto &inj = rtc_config.inj_signal;

        // Always target Tip/Tilt branch
        inj.enabled   = true;
        inj.branch    = "LO";              // Tip/Tilt
        inj.apply_to  = apply_to;          // "setpoint" (recommended first) or "command"
        inj.amplitude = A;

        // Preserve user's current basis/basis_index; they may point to a "tt" basis already.
        // If you prefer to force actuator (zonal) basis, uncomment next two lines:
        // inj.basis       = "zonal";
        // inj.basis_index = 0; // <-- user can change via commander if desired

        // Common defaults
        inj.latency_frames = 0;            // do not add artificial delay for ID
        inj.hold_frames    = 1;            // update every frame by default
        inj.phase_deg      = 0.0;
        inj.duty           = 0.5;          // for square if ever used
        // IMPORTANT: we do NOT set t_start_s / t_stop_s here because rtc.cpp's now_s()
        // timestamps are internal to the RTC; set those via existing field setters if needed.

        if (method == "chirp") {
            inj.waveform = "chirp";
            inj.chirp_f0 = 0.5;            // Hz (start)
            inj.chirp_f1 = 400.0;          // Hz (end) — leave guard to 500 Hz Nyquist at 1 kHz
            inj.chirp_T  = 60.0;           // seconds sweep (good resolution ~1/60 Hz)
            inj.hold_frames = 1;           // dense spectrum to Nyquist
            // Leave t_start_s/t_stop_s unchanged (user can gate via commander)
        }
        else if (method == "prbs") {
            inj.waveform  = "prbs";
            inj.prbs_seed = inj.prbs_seed ? inj.prbs_seed : 0xBEEF; // keep user seed if set
            // For TT focus (≤250 Hz) you can decimate the chip rate a bit:
            inj.hold_frames = 2;           // chip rate = fs/hold = 500 Hz (good energy ≤ ~250 Hz)
            // If you want flatter up to Nyquist, use hold_frames=1
        }
        else if (method == "sine") {
            inj.waveform = "sine";
            inj.freq_hz  = 10.0;           // a sensible default near typical TT crossover
            inj.hold_frames = 1;
            // phase_deg already 0.0
        }
    }

    // Reset runtime so PRBS LFSR, chirp phase, sample-and-hold, etc. start cleanly.
    reset_signal_injection_runtime();

    std::cout << "[inj] TT configured: method=" << method
              << ", A=" << A
              << ", apply_to=" << apply_to
              << " (branch=LO)\n";
    return true;
}



// Commander wrapper: configure Tip/Tilt (LO) injection from a JSON payload.
// Accepts either an array form: ["chirp|prbs|sine", amplitude, "setpoint|command"]
// or an object form: {"method":"prbs","amplitude":0.02,"apply_to":"setpoint"}
nlohmann::json tt_inj(nlohmann::json args) {
    using json = nlohmann::json;

    // Defaults (safe)
    std::string method   = "chirp";
    double      amplitude = 0.02;
    std::string apply_to = "setpoint";

    try {
        if (args.is_array()) {
            if (args.size() < 1 || args.size() > 3) {
                return json{{"ok", false}, {"error",
                    R"(usage: ["chirp|prbs|sine", amplitude?, "setpoint|command"?])"}};
            }
            method = args.at(0).get<std::string>();
            if (args.size() >= 2) amplitude = args.at(1).get<double>();
            if (args.size() >= 3) apply_to  = args.at(2).get<std::string>();
        } else if (args.is_object()) {
            if (args.contains("method"))    method    = args.at("method").get<std::string>();
            if (args.contains("amplitude")) amplitude = args.at("amplitude").get<double>();
            if (args.contains("apply_to"))  apply_to  = args.at("apply_to").get<std::string>();
        } else {
            return json{{"ok", false}, {"error", "args must be array or object"}};
        }
    } catch (const std::exception& e) {
        return json{{"ok", false}, {"error", std::string("bad args: ") + e.what()}};
    }

    const bool ok = configure_tt_injection(method, amplitude, apply_to);
    return json{
        {"ok", ok},
        {"method", method},
        {"amplitude", amplitude},
        {"apply_to", apply_to},
        {"branch", "LO"}
    };
}



// ---------- Probe method wrapper + FITS writer ----------

static int save_probe_result_fits(const nlohmann::json& jr, const std::string& filepath) {
    using json = nlohmann::json;

    if (!jr.value("ok", false)) {
        std::cerr << "[probe] cannot save: result not ok.\n";
        return -1;
    }
    if (!jr.contains("probes") || !jr["probes"].is_array()) {
        std::cerr << "[probe] cannot save: no 'probes' array.\n";
        return -2;
    }

    const auto& probes = jr["probes"];
    const long nrows = static_cast<long>(probes.size());
    if (nrows <= 0) {
        std::cerr << "[probe] nothing to save: zero probes.\n";
        return -3;
    }

    // Infer dimensions from first probe
    if (!probes[0].contains("samples") || !probes[0]["samples"].is_array())
        throw std::runtime_error("[probe] malformed JSON: probes[0].samples missing/invalid");

    const auto& s0   = probes[0]["samples"];
    const long NSAMP = static_cast<long>(s0.size());
    if (NSAMP <= 0) throw std::runtime_error("[probe] first probe has zero samples");

    const long VLEN = static_cast<long>(s0[0].size());  // vector length per sample
    if (VLEN <= 0) throw std::runtime_error("[probe] first sample has zero length");

    // Validate uniformity
    for (long i = 0; i < nrows; ++i) {
        const auto& ps = probes[static_cast<size_t>(i)]["samples"];
        if (!ps.is_array() || static_cast<long>(ps.size()) != NSAMP)
            throw std::runtime_error("[probe] non-uniform number of samples across probes");
        for (long k = 0; k < NSAMP; ++k) {
            if (!ps[static_cast<size_t>(k)].is_array() ||
                static_cast<long>(ps[static_cast<size_t>(k)].size()) != VLEN)
                throw std::runtime_error("[probe] non-uniform vector length in samples");
        }
    }

    fitsfile* fptr = nullptr;
    int status = 0;

    auto cleanup = [&]() {
        if (fptr) {
            int s2 = 0;
            fits_close_file(fptr, &s2);
            fptr = nullptr;
        }
    };

    try {
        // Ensure parent directory exists
        try {
            std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
        } catch (...) { /* best effort */ }

        if (fits_create_file(&fptr, ("!" + filepath).c_str(), &status)) {
            fits_report_error(stderr, status);
            cleanup();
            return status ? status : -99;
        }

        // Column layout:
        // 1 IDX (J)     actuator/mode index
        // 2 AMP (D)     amplitude
        // 3 VLEN (J)    vector length per sample
        // 4 NSAMP (J)   number of sequential samples
        // 5 DATA ((VLEN*NSAMP)D) flattened [sample-major]
        const int ncols = 5;
        char* ttype[ncols] = {
            const_cast<char*>("IDX"),
            const_cast<char*>("AMP"),
            const_cast<char*>("VLEN"),
            const_cast<char*>("NSAMP"),
            const_cast<char*>("DATA")
        };

        std::string data_form = std::to_string(VLEN * NSAMP) + "D";
        char tform0[8]  = "1J";
        char tform1[8]  = "1D";
        char tform2[8]  = "1J";
        char tform3[8]  = "1J";
        char tform4[32] = {0};
        std::snprintf(tform4, sizeof(tform4), "%s", data_form.c_str());

        char* tform[ncols] = { tform0, tform1, tform2, tform3, tform4 };
        char* tunit[ncols] = {
            const_cast<char*>(""), const_cast<char*>(""),
            const_cast<char*>(""), const_cast<char*>(""),
            const_cast<char*>("")
        };

        if (fits_create_tbl(fptr, BINARY_TBL, nrows, ncols, ttype, tform, tunit,
                            const_cast<char*>("PROBES"), &status)) {
            fits_report_error(stderr, status);
            cleanup();
            return status ? status : -99;
        }

        // Header metadata
        auto putstr = [&](const char* key, const std::string& val) {
            char buf[2880];
            std::snprintf(buf, sizeof(buf), "%s", val.c_str());
            fits_update_key(fptr, TSTRING, const_cast<char*>(key), buf, nullptr, &status);
        };
        auto putint = [&](const char* key, long v) {
            long vv = v;
            fits_update_key(fptr, TLONG, const_cast<char*>(key), &vv, nullptr, &status);
        };

        // --- Camera + Boxcar headers (legacy BINTABLE) ---
        try {
            double gain = static_cast<double>(get_float_cam_param("gain raw"));
            double fps  = static_cast<double>(get_float_cam_param("fps raw"));
            fits_update_key(fptr, TDOUBLE, const_cast<char*>("GAINRAW"),
                            &gain, const_cast<char*>("camera gain (raw units)"), &status);
            fits_update_key(fptr, TDOUBLE, const_cast<char*>("FPSRAW"),
                            &fps,  const_cast<char*>("camera fps (raw units)"),  &status);
        } catch (...) {
            //  ignore if camera params unavailable
            std::cerr << "[probe] camera params unavailable; skipping GAINRAW/FPSRAW\n";

        }

        try {
            // If your boxcar is a *length* (integer), prefer NBOXCAR as TLONG:
            long nbox = 0;
            try {
                // If global_boxcar.load() returns a numeric window length:
                nbox = static_cast<long>(std::llround(global_boxcar.load()));
            } catch (...) {
                // ignore and output warning
                std::cerr << "[probe] boxcar unavailable; skipping NBOXCAR\n";
            }
            fits_update_key(fptr, TLONG, const_cast<char*>("NBOXCAR"),
                            &nbox, const_cast<char*>("boxcar window length (samples)"), &status);

            // Optionally also store a floating value (if you have one):
            // double gbox = static_cast<double>(global_boxcar.load());
            // fits_update_key(fptr, TDOUBLE, const_cast<char*>("GBOXCAR"),
            //                 &gbox, const_cast<char*>("boxcar value"), &status);
        } catch (...) {
            //  ignore if camera params unavailable
            std::cerr << "[probe] boxcar unavailable; skipping NBOXCAR\n";

        }

        putstr("BASIS",    jr.value("basis",      std::string("")));
        putstr("STATE",    jr.value("state",      std::string("")));
        putstr("FIELD",    jr.value("field",      std::string("")));
        putint("N",        jr.value("N",          0));
        putint("DEADTIME", jr.value("deadtime_ms",0));
        putint("USEDCAP",  jr.value("used_capacity", 0));

        // Timestamp
        {
            auto now = std::chrono::system_clock::now();
            std::time_t tt = std::chrono::system_clock::to_time_t(now);
            char tbuf[64];
            std::strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&tt));
            putstr("DATE", tbuf);
        }

        if (status) {
            fits_report_error(stderr, status);
            cleanup();
            return status ? status : -99;
        }

        // Write rows
        for (long r = 0; r < nrows; ++r) {
            const auto& pr = probes[static_cast<size_t>(r)];
            int idx    = pr.value("idx", 0);
            double amp = pr.value("amp", 0.0);

            if (fits_write_col(fptr, TINT,    1, r+1, 1, 1, &idx,  &status) ||
                fits_write_col(fptr, TDOUBLE, 2, r+1, 1, 1, &amp,  &status) ||
                fits_write_col(fptr, TINT,    3, r+1, 1, 1, (void*)&VLEN,  &status) ||
                fits_write_col(fptr, TINT,    4, r+1, 1, 1, (void*)&NSAMP, &status)) {
                fits_report_error(stderr, status);
                cleanup();
                return status ? status : -99;
            }

            std::vector<double> flat;
            flat.reserve(static_cast<size_t>(VLEN * NSAMP));
            const auto& samples = pr["samples"];
            for (long k = 0; k < NSAMP; ++k) {
                const auto& v = samples[static_cast<size_t>(k)];
                for (long j = 0; j < VLEN; ++j)
                    flat.push_back(v[static_cast<size_t>(j)].get<double>());
            }

            if (fits_write_col(fptr, TDOUBLE, 5, r+1, 1, VLEN*NSAMP, flat.data(), &status)) {
                fits_report_error(stderr, status);
                cleanup();
                return status ? status : -99;
            }
        }

        if (fits_close_file(fptr, &status)) {
            fits_report_error(stderr, status);
            fptr = nullptr;
            return status ? status : -99;
        }
        fptr = nullptr;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[probe] exception while saving: " << e.what() << "\n";
        cleanup();
        return status ? status : -99;
    }
}


// // baldr.cpp
// bool save_probe_result_fits(const std::string& path,
//                             const ProbeResult& R,
//                             const std::vector<std::string>& fields)
// {
//     // Probe uses the last R.n_steps samples from telemetry buffers.
//     auto extractor = [&](std::string_view name, Eigen::MatrixXd& M, std::string& why) {
//         return telem_extract_matrix_lastN(rtc_config.telem, telemetry_mutex, name, R.n_steps, M, why);
//     };
//     return write_fields_to_fits(path, fields, extractor, static_cast<long>(R.n_steps));
// }

// // Optional single-field wrapper for backwards compatibility
// bool save_probe_result_fits(const std::string& path,
//                             const ProbeResult& R,
//                             std::string_view field)
// {
//     return save_probe_result_fits(path, R, std::vector<std::string>{std::string(field)});
// }

bool telem_extract_matrix_lastN(const bdr_telem& telem,
                                std::mutex& telemetry_mutex,
                                std::string_view field,
                                std::size_t N,
                                Eigen::MatrixXd& M,
                                std::string& why);

// New N-based multi-field saver
bool save_probe_result_fits(const std::string& path,
                            std::size_t N,
                            const std::vector<std::string>& fields)
{
    auto extractor = [&](std::string_view name, Eigen::MatrixXd& M, std::string& why) {
        return telem_extract_matrix_lastN(rtc_config.telem, telemetry_mutex, name, N, M, why);
    };
    return write_fields_to_fits(path, fields, extractor, static_cast<long>(N));
}

///!!!! 
// Build default args for a named method of applying aberrations/offsets in closed or open loop and recording telemetry.
static nlohmann::json build_probe_args_for_method(const std::string& method) {
    using json = nlohmann::json;

    // method auto naming convention 
    // cust_<loop_state>_<basis>_<signal>_<samples>
    // then the code will infer the loop state basis signal and samples from this input string convention

    if (method == "method_1") {
        // testing 123
        // probe_interaction_data [[[5,0.05],[12,-0.03]],"zernike","OL","signal",3,5000]
        // aa = util.get_DM_command_in_2D( np.mean( d[1].data['DATA'][0].reshape(3,-1), axis=0 ) )
        json aberr = json::array();
        aberr.push_back(json::array({65,  0.1}));
        aberr.push_back(json::array({2, -0.1}));
        return json::array({ aberr, "zonal", "OL", "all", 3, 5000 });
    }

    if (method == "I2A") {
        // probes 4 inner corners to solve DM registration matrix
        // Note the convention here my be 12x12 (144 for zonal) not 140!
        //In [2]: DM_registration.get_inner_square_indices(outer_size=12, inner_offset=4)
        // Out[2]: [np.int64(50), np.int64(53), np.int64(86), np.int64(89)]

        // In [3]: DM_registration.get_inner_square_indices(outer_size=12, inner_offset=3)
        // Out[3]: [np.int64(37), np.int64(42), np.int64(97), np.int64(102)]

        // In [4]: DM_registration.get_inner_square_indices(outer_size=12, inner_offset=2)
        // Out[4]: [np.int64(24), np.int64(31), np.int64(108), np.int64(115)]
        
        // NEED TO TEST METHOD
        json aberr = json::array();

        for (int iii : {50, 53, 86, 89}) { // the four inner corners (4 rows in)
            aberr.push_back(json::array({iii,  0.04}));
            aberr.push_back(json::array({iii, -0.04}));
        }
        return json::array({ aberr, "zonal", "OL", "img", 4, 5000 });
    }




    if (method == "TT_OL_img") {

        json aberr = json::array();

        for (int iii : {0,1}) { // the four inner corners (4 rows in)
            aberr.push_back(json::array({iii,  0.03}));
            aberr.push_back(json::array({iii, -0.03}));
        }
        return json::array({ aberr, "zonal", "OL", "img", 10, 5000 });
    }



    if (method == "I2A_CL"){
        
        // building interaction matrix for Baldr 
        json aberr = json::array();

        for (int iii : {50, 53, 86, 89}) { // the four inner corners (4 rows in)
            aberr.push_back(json::array({iii,  0.04}));
            aberr.push_back(json::array({iii, -0.04}));
        }
        return json::array({ aberr, "zonal", "CL", "img", 10, 5000 });
    }
    if (method == "IM_OL"){
        
        // building interaction matrix for Baldr 
        json aberr = json::array();

        for (int iii = 0; iii < 140; ++iii){
            aberr.push_back(json::array({iii,  0.04}));
            aberr.push_back(json::array({iii, -0.04}));
        }

        return json::array({ aberr, "zonal", "OL", "img", 3, 5000 });

    } 

    if (method == "IM_CL"){
        
        // building interaction matrix for Baldr 
        json aberr = json::array();

        for (int iii = 0; iii < 140; ++iii){
            aberr.push_back(json::array({iii,  0.04}));
            aberr.push_back(json::array({iii, -0.04}));
        }

        return json::array({ aberr, "zonal", "CL", "img", 8, 5000 });

    } 


    //+++++++++++++++++++++++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++
    // Add more named methods here…
    // else if (method == "method_2") { … }
    //+++++++++++++++++++++++++++++++++++++
    //+++++++++++++++++++++++++++++++++++++
    throw std::invalid_argument("Unknown probe method: " + method);
}

// Commander-facing wrapper:
// Accepts: ["method_1", "/path/or/dir/optional"]
//      e.g ["method_1","/home/rtc/Downloads/test.fits"]
// TO do : standard naming convention e.g. last _ seperated string is basis and last is telemetry
// I2A, IM, I0
//// newer function to check if we want to save all fields in probe, otherwise uses legacu functions for single field
static nlohmann::json run_probe_method(nlohmann::json args) {
    using json = nlohmann::json;

    std::string method;
    std::string out_path;  // optional

    // --- Parse ---
    try {
        if (args.is_array()) {
            if (args.size() < 1 || args.size() > 2)
                return json{{"ok", false}, {"error", "usage: [\"method_name\", optional_path]"}};
            method = args.at(0).get<std::string>();
            if (args.size() == 2) out_path = args.at(1).get<std::string>();
        } else if (args.is_object()) {
            if (!args.contains("method"))
                return json{{"ok", false}, {"error", "missing 'method'"}};
            method = args.at("method").get<std::string>();
            if (args.contains("path")) out_path = args.at("path").get<std::string>();
        } else {
            return json{{"ok", false}, {"error", "bad args: expected array or object"}};
        }
    } catch (const std::exception& e) {
        return json{{"ok", false}, {"error", std::string("parse: ") + e.what()}};
    }

    // --- Build default probe args for the selected method ---
    json probe_args;
    try {
        probe_args = build_probe_args_for_method(method);
    } catch (const std::exception& e) {
        return json{{"ok", false}, {"error", e.what()}};
    }

    // --- Run the probe ---
    json result;
    try {
        result = probe_interaction_data(probe_args);
    } catch (const std::exception& e) {
        return json{{"ok", false}, {"error", std::string("probe_interaction_data threw: ") + e.what()}};
    }
    if (!result.value("ok", false)) {
        return json{{"ok", false}, {"error", "probe_interaction_data failed"}, {"detail", result}};
    }

    // --- Resolve output path we want to return ---
    std::string filepath;
    try {
        if (out_path.empty()) {
            // default base: telem_save_path (if set) or fallback dir
            auto now = std::chrono::system_clock::now();
            std::time_t tt = std::chrono::system_clock::to_time_t(now);
            char tbuf[32];
            std::strftime(tbuf, sizeof(tbuf), "%Y%m%d_%H%M%S", std::localtime(&tt));

            std::filesystem::path base = telem_save_path.empty()
                ? std::filesystem::path("/usr/local/var/baldr/probes")
                : std::filesystem::path(telem_save_path);

            std::filesystem::create_directories(base);
            filepath = (base / ("probe_" + method + "_" + std::string(tbuf) + ".fits")).string();
        } else {
            std::filesystem::path p(out_path);
            if (p.has_extension()) {
                // Treat as full filename
                std::filesystem::create_directories(p.parent_path());
                filepath = p.string();
            } else {
                // Treat as directory
                std::filesystem::create_directories(p);
                auto now = std::chrono::system_clock::now();
                std::time_t tt = std::chrono::system_clock::to_time_t(now);
                char tbuf[32];
                std::strftime(tbuf, sizeof(tbuf), "%Y%m%d_%H%M%S", std::localtime(&tt));
                filepath = (p / ("probe_" + method + "_" + std::string(tbuf) + ".fits")).string();
            }
        }
    } catch (const std::exception& e) {
        return json{{"ok", false}, {"error", std::string("path resolution: ") + e.what()}};
    }

    // --- Save / Move depending on mode ---
    const std::string mode = result.value("mode", std::string{});
    if (mode == "multi-field") {
        // probe_interaction_data already wrote a FITS at result["outfile"].
        const std::string src = result.value("outfile", std::string{});
        if (src.empty()) {
            return json{{"ok", false}, {"error", "multi-field result missing 'outfile'"}};
        }

        try {
            if (src != filepath) {
                std::error_code ec;
                std::filesystem::create_directories(std::filesystem::path(filepath).parent_path(), ec);
                // Try rename (atomic move), fall back to copy+remove
                std::filesystem::rename(src, filepath, ec);
                if (ec) {
                    ec.clear();
                    std::filesystem::copy_file(src, filepath,
                        std::filesystem::copy_options::overwrite_existing, ec);
                    if (ec) {
                        return json{{"ok", false},
                                    {"error", "failed to move/copy outfile to target path"},
                                    {"src", src}, {"dst", filepath}};
                    }
                    std::filesystem::remove(src); // best-effort cleanup; ignore errors
                }
            }
        } catch (const std::exception& e) {
            return json{{"ok", false},
                        {"error", std::string("finalize (multi-field) failed: ") + e.what()},
                        {"src", src}, {"dst", filepath}};
        }

        // Build success payload (include fields if present)
        json out = {
            {"ok", true},
            {"method", method},
            {"path", filepath},
            {"n_probes", result.value("n_probes", 0)},
            {"basis", result.value("basis", "")},
            {"state", result.value("state", "")},
            {"mode", "multi-field"}
        };
        if (result.contains("fields")) out["fields"] = result["fields"];
        if (result.contains("N"))      out["N"]      = result["N"];
        return out;
    }

    // Legacy single-field path: write BINTABLE from JSON
    int st = save_probe_result_fits(result, filepath);
    if (st != 0) {
        return json{
            {"ok", false},
            {"error", "save_probe_result_fits failed"},
            {"status", st},
            {"path", filepath}
        };
    }

    return json{
        {"ok", true},
        {"method", method},
        {"path", filepath},
        {"n_probes", result.value("n_probes", 0)},
        {"basis", result.value("basis", "")},
        {"state", result.value("state", "")},
        {"field", result.value("field", "")},
        {"N", result.value("N", 0)},
        {"mode", "single-field"}
    };
}


/// before multi field saves we used this function 
// static nlohmann::json run_probe_method(nlohmann::json args) {
//     using json = nlohmann::json;

//     std::string method;
//     std::string out_path;  // optional

//     // --- Parse ---
//     try {
//         if (args.is_array()) {
//             if (args.size() < 1 || args.size() > 2)
//                 return json{{"ok", false}, {"error", "usage: [\"method_name\", optional_path]"}};
//             method = args.at(0).get<std::string>();
//             if (args.size() == 2) out_path = args.at(1).get<std::string>();
//         } else if (args.is_object()) {
//             if (!args.contains("method"))
//                 return json{{"ok", false}, {"error", "missing 'method'"}};
//             method = args.at("method").get<std::string>();
//             if (args.contains("path")) out_path = args.at("path").get<std::string>();
//         } else {
//             return json{{"ok", false}, {"error", "bad args: expected array or object"}};
//         }
//     } catch (const std::exception& e) {
//         return json{{"ok", false}, {"error", std::string("parse: ") + e.what()}};
//     }

//     // --- Build default probe args for the selected method ---
//     json probe_args;
//     try {
//         probe_args = build_probe_args_for_method(method);
//     } catch (const std::exception& e) {
//         return json{{"ok", false}, {"error", e.what()}};
//     }


//     // --- Run the probe ---
//     json result;
//     try {
//         result = probe_interaction_data(probe_args);
//     } catch (const std::exception& e) {
//         return json{{"ok", false}, {"error", std::string("probe_interaction_data threw: ") + e.what()}};
//     }
//     if (!result.value("ok", false)) {
//         return json{{"ok", false}, {"error", "probe_interaction_data failed"}, {"detail", result}};
//     }

//     // --- Resolve output path ---
//     std::string filepath;
//     try {
//         if (out_path.empty()) {
//             // default base: telem_save_path (if set) or fallback dir
//             auto now = std::chrono::system_clock::now();
//             std::time_t tt = std::chrono::system_clock::to_time_t(now);
//             char tbuf[32];
//             std::strftime(tbuf, sizeof(tbuf), "%Y%m%d_%H%M%S", std::localtime(&tt));

//             std::filesystem::path base = telem_save_path.empty()
//                 ? std::filesystem::path("/usr/local/var/baldr/probes")
//                 : std::filesystem::path(telem_save_path);

//             std::filesystem::create_directories(base);
//             filepath = (base / ("probe_" + method + "_" + std::string(tbuf) + ".fits")).string();
//         } else {
//             std::filesystem::path p(out_path);
//             if (p.has_extension()) {
//                 // Treat as full filename
//                 std::filesystem::create_directories(p.parent_path());
//                 filepath = p.string();
//             } else {
//                 // Treat as directory
//                 std::filesystem::create_directories(p);
//                 auto now = std::chrono::system_clock::now();
//                 std::time_t tt = std::chrono::system_clock::to_time_t(now);
//                 char tbuf[32];
//                 std::strftime(tbuf, sizeof(tbuf), "%Y%m%d_%H%M%S", std::localtime(&tt));
//                 filepath = (p / ("probe_" + method + "_" + std::string(tbuf) + ".fits")).string();
//             }
//         }
//     } catch (const std::exception& e) {
//         return json{{"ok", false}, {"error", std::string("path resolution: ") + e.what()}};
//     }

//     // --- Save to FITS ---
//     int st = save_probe_result_fits(result, filepath);
//     if (st != 0) {
//         return json{
//             {"ok", false},
//             {"error", "save_probe_result_fits failed"},
//             {"status", st},
//             {"path", filepath}
//         };
//     }

//     return json{
//         {"ok", true},
//         {"method", method},
//         {"path", filepath},
//         {"n_probes", result.value("n_probes", 0)},
//         {"basis", result.value("basis", "")},
//         {"state", result.value("state", "")},
//         {"field", result.value("field", "")},
//         {"N", result.value("N", 0)}
//     };
// }

//////////////////


// input
// aberrations: list of tuples of [(indicies, amplitudes)..]
// basis = str (name of modal basis to apply)
// state: string (OL | CL)
// signal: string (what signal from the RTC telemetry to record)
// duration_per_cmd : int (number of frames to capture per command)
// dead_time = (optional) int (number of frames to wait before recording)

// output a list of list of signals obtained for each command 

// pseudo code 

// basis = string 
// res = init structure to hold results  
// for a in aberrations:
//     idx, amp = a 
//     if state.strip() == 'OL':
//         //offset DM in open loop 
//         ol_set_mode ["HO",<basis>,<idx>,<amp>]
//     elif state.strip() == 'CL'
//         //offset DM in closed loop (using controller set_point)
//         mode = amp * read(basis)[idx]
//         ctrl_set ["HO","set_point",mode]
//     else:
//         raise typeError( "Invalid input state, valid states are CL or OL")
//     wait deadtime 
//     ims = record images 
//     res.append( ims ) 

// return res

// ## To do 
// # [x] load basis functionality (store basis in toml/csv/fits? where to store, usr/etc/baldr? what normalization convention?)
// # independant of telemetry buffer size, copy N independant frames even when N> or N<buffer size.  

// ## questions
// # what structure to store results in (keeping in mind it could be thousands of a 32x32 image)
// # second function will get these results and save them. 


// static json probe_interaction_data( list of tuples or similar ::aberration, std::str state, std::str field, int N, int deadtime_ms=1)
// for a in aberrations:
//     idx, amp = a 
//     if state.strip() == 'OL':
//         //offset DM in open loop 
//         ol_set_mode ["HO",basis,idx,amp]
//     elif state.strip() == 'CL'
//         //offset DM in closed loop (using controller set_point)
//                 // ---- Lookup basis ----
//         auto modes_ptr = dm::get_basis(basis); // lowercase lookup inside
//         // if (!modes_ptr)
//         //     return json{{"ok", false}, {"error", "basis not found"}, {"basis", basis}};

//         const auto& modes = *modes_ptr;
//         // if (idx_ll >= static_cast<long long>(modes.size()))
//         //     return json{{"ok", false}, {"error", "index out of range"}, {"n_modes", (long)modes.size()}};

//         const Eigen::VectorXd& mode144 = modes[static_cast<size_t>(idx_ll)];

//         mode144 *= amp 
//         mode = dm144_to_140(mode144)

//         ctrl_set ["HO","set_point",mode]
//     else:
//         raise typeError( "Invalid input state, valid states are CL or OL")
    
//     sleep deadtime_ms
//     ims = capture_telem_list( field, N, deadtime_ms=0)
//     res.append( ims ) 


COMMANDER_REGISTER(m)
{
    using namespace commander::literals;

    // You can register a function or any other callable object as
    // long as the signature is deductible from the type.
    // m.def("load_configuration", load_configuration, "Load a configuration file. Return true if successful.", 
    //     "filename"_arg="def.toml");
    m.def("status", []() {
        Status s = get_status();
        return nlohmann::json{
            {"TT_state",        s.TT_state},
            {"HO_state",        s.HO_state},
            {"mode",            s.mode},
            {"phasemask",       s.phasemask},
            {"frequency",       s.frequency},
            {"configured",      s.configured},
            {"ctrl_type",       s.ctrl_type},
            {"config_file",     s.config_file},
            {"inj_enabled",     s.inj_enabled},
            {"auto_loop",       s.auto_loop},
            {"close_on_strehl", s.close_on_strehl},
            {"open_on_strehl",  s.open_on_strehl},
            {"close_on_snr",    s.close_on_snr},
            {"open_on_snr",     s.open_on_snr},
            {"TT_offsets",      s.TT_offsets}
        };
    });
    //send_cam_command ["set gain 1"]
    m.def("send_mds_command", send_mds_command,
        "Send a raw string command to the MultiDeviceServer over ZMQ.\n"
        "Usage: send_mds_command [\"your command\"] e.g. send_mds_command [\"off SBB\"]",
        "args"_arg);

    //send_mds_command ["off SBB"]
    m.def("send_cam_command", send_cam_command,
        "Send a raw string command to the Camera Server (FLI) over ZMQ.\n"
        "Usage: send_cam_command [\"your command\"]. e.g. send_cam_command [\"set gain 1\"]",
        "args"_arg);
        
    // outdated , reduction is done via cred1 camera
    // m.def("capture_dark_and_bias", capture_dark_and_bias,
    //         "Capture a new bias and dark frame, update reduction products, and update DM-space maps.");
            
    m.def("stop_baldr", stop_baldr,
          "stop the baldr rtc loop","mode"_arg);

    m.def("close_baldr_LO", close_baldr_LO,
          "close the LO servo loop for baldr","mode"_arg);

    m.def("open_baldr_LO", open_baldr_LO,
          "open the LO servo loop for baldr - resetting gains and flatten DM","mode"_arg);

    m.def("close_all", close_all,
          "close LO and HO servo loop for baldr","mode"_arg);

    m.def("open_all", open_all,
          "open LO and HO servo loop for baldr - resetting gains and flatten DM","mode"_arg);

    m.def("close_baldr_HO", close_baldr_HO,
          "close the HO servo loop for baldr","mode"_arg);

    m.def("open_baldr_HO", open_baldr_HO,
          "open the HO servo loop for baldr - resetting gains and flatten DM","mode"_arg);

    m.def("save_telemetry", save_telemetry,
          "dump telemetry in the circular buffer to file","mode"_arg);

    m.def("set_telem_save_path", set_telem_save_path,
        "Set the directory to save telemetry files. Usage: set_telem_save_path [\"/your/save/path/\"]",
        "args"_arg);
        
    m.def("set_telem_capacity", set_telem_capacity,
          "Set the capacity for all telemetry ring buffers (how many samples we hold). e.g: set_telem_capacity [200]",
          "args"_arg);


    m.def("configure_burst", cmd_configure_burst,  "Update the rolling burst window (size and dt) for non-destructive read mode slope estimates", "args"_arg);


    // uncomment and build July 2025 AIV
    m.def("change_boxcar", change_boxcar,  "Update the number of signal samples from telem ring buffer to weight an average. Only applied for values > 1", "args"_arg);


    m.def("I0_update", I0_update,
            "Update I0_dm_runtime by averaging the telemetry img_dm buffer (no arguments required).");

    m.def("I0_update_lucky", [](){
        I0_update_lucky(95.0, 10);  // default: top 10% frames, require >=10 frames available
    });

    m.def("N0_update", N0_update,
            "Update N0 reference: averages img telemetry, corrects exterior pixels, and projects to DM space");
            

    m.def("dark_update", dark_update,
      "Average the telemetry RAW img buffer and store to reduction.dark (ADU).");

    m.def("set_telem_save_format", set_telem_save_format,
          "Update the telemetry output format. Acceptable values: \"json\", \"fits\".\n"
          "Usage: set_telem_format [\"json\"] or set_telem_format [\"fits\"]",
          "args"_arg);


    m.def("pause_baldr_rtc", [](){
        pauseRTC();
        return std::string("RTC paused.");
    }, "Pause the RTC loop.\n\n");

    m.def("resume_baldr_rtc", [](){
        resumeRTC();
        return std::string("RTC resumed.");
    }, "Resume the RTC loop.\n\n");


    // commented out while upgrading controller class 20-8-25
    // //update_pid_param ["LO","kp", "all",0.1] or update_pid_param ["HO","kp","1,3,5,42",0.1] to update gains of particular mode indicies  (1,3,5,42) to 0.1
    // m.def("update_pid_param", update_pid_param,
    //       "Update a PID gain vector or set_point, upper_limit or lower_limit. Parameters: [mode, parameter, indices, value].\n"
    //       "  - mode: LO or HO"
    //       "  - gain_type: \"kp\", \"ki\", or \"kd\"\n"
    //       "  - indices: a comma-separated list of indices or \"all\"\n"
    //       "  - value: the new gain value (number)\n"
    //       "e.g. (use double quotation) update_pid_param [''LO'',''kp'',''all'', 0.0] or update_pid_param [''HO'',''kp'',''1,3,5,42'',0.1] or ",
    //       "args"_arg);




    // //print_pid_attribute ["HO","ki"], print_pid_attribute ["LO","kp"]
    // m.def("print_pid_attribute", print_pid_attribute,
    //       "Print the specified attribute of a PID controller.\n"
    //       "Usage: print_pid_attribute [controller, attribute]\n"
    //       "  - controller: \"LO\" or \"HO\"\n"
    //       "  - attribute: one of \"kp\", \"ki\", \"kd\", \"lower_limits\", \"upper_limits\", \"set_point\", "
    //       "\"output\", \"integrals\", or \"prev_errors\"",
    //       "args"_arg);

    //build_interaction_matrix [0.05, 10, 0.1, "dm", "/home/asg/Music/IM_test.fits"]
    m.def("build_interaction_matrix",
        [](json args) {
            if (!args.is_array() || args.size() > 5) {
                return json{{"error", "Usage: build_interaction_matrix [poke_amp, num_iterations, sleep_seconds, signal_space, output_filename]"}};
            }

            double poke_amp = args.size() > 0 ? args[0].get<double>() : 0.05;
            int num_iterations = args.size() > 1 ? args[1].get<int>() : 10;
            double sleep_seconds = args.size() > 2 ? args[2].get<double>() : 0.01;
            std::string signal_space = args.size() > 3 ? args[3].get<std::string>() : "dm";
            std::string output_filename = args.size() > 4 ? args[4].get<std::string>() : "";

            build_interaction_matrix(poke_amp, num_iterations, sleep_seconds, signal_space, output_filename);

            return json{{"status", "interaction matrix construction complete"}};
            },
            "Builds interaction matrix: build_interaction_matrix [poke_amp, num_iter, sleep_s, signal_space, output_filename]",
            "args"_arg
    );

    //e.g. reload_config ["new_config.toml", "H5"]
    m.def("reload_config", reload_config,
          "Reload a new configuration file with a specified phasemask. It must be .toml \n"
          "from the baldr calibration pipeline. Also assumes the same beam_id as is currently configured.\n"
          "Usage: reload_config [\"new_filename.toml\", \"new_mask\"]",
          "args"_arg);

    //test in TTR115.0035 (not tested yet)
    m.def("poll_telem_vector", poll_telem_vector,
        "poll the second last entry to baldr's rolling telemetry buffers for the given field (must be a vector).\n"
        "Usage: poll_telem_vector u_HO\n",
        "telem fields : img, img_dm, signal, e_LO, u_LO, e_HO, u_HO, c_LO, c_HO"
        "args"_arg);

    // test in TTR115.0035 (not tested yet)
    m.def("poll_telem_scalar", poll_telem_scalar,
        "poll the second last entry to baldr's rolling telemetry buffers for the given field \n"
        "Usage: poll_telem_vector snr",
        "args"_arg);

    m.def("ol_set_actuator", ol_set_actuator,
        "set an open loop DM offset on one actuator.  Branch is HO|LO, idx is index in vector and value to set. e.g. ol_set_actuator [\"HO\", 65, 0.1] ",
        "branch"_arg, "idx"_arg, "value"_arg);

    m.def("ol_set_mode",
      [](json args){ return ol_set_mode(args); },
      "ol_set_mode — set/add an open-loop offset in HO or LO using a basis mode; usage: [\"HO|LO\",\"<basis>\",<index>,<amplitude>,(optional)<accumulate_bool>] (accumulate defaults false).");

    m.def("clear_openloop_offsets",clear_openloop_offsets,
        "clears the open loop DM offsets");
        
    m.def("list_rtc_fields", list_rtc_fields,
        "List exposed runtime config fields and types.",
        "args"_arg);

    m.def("get_rtc_field", get_rtc_field,
        "Get one runtime config field by path (string). Usage: get_rtc_field \"gain\"",
        "args"_arg);

    m.def("set_rtc_field", set_rtc_field,
        "Set one runtime config field by path and JSON value. Usage: set_rtc_field set_rtc_field \"gain\",2",
        "args"_arg);

    m.def("ctrl_list", ctrl_list,
        "List controller parameters and types (LO|HO)",
        "args"_arg);

    m.def("ctrl_get", ctrl_get,
        "Get a controller parameter (vector/matrix/string)",
        "args"_arg);

    m.def("ctrl_set", ctrl_set,
        "Set a controller parameter (vector or matrix). eg. ctrl_set [\"LO\",\"set_point\",[0,0.15]]",
        "args"_arg);

    /////    
    // below are utils that allow relative moves and/or updating all or multiple parameters at once without necessarily writing out full vector!
    /////
    //update_pid_param ["LO","kp", "all",0.1] or update_pid_param ["HO","kp","1,3,5,42",0.1] to update gains of particular mode indicies  (1,3,5,42) to 0.1
    m.def("update_ctrl_param", update_ctrl_param,
          "Update a controller (e.g. PID) gain vector or set_point, upper_limit or lower_limit. Parameters: [mode, parameter, indices, value].\n"
          "  - mode: LO or HO"
          "  - parameter: \"kp\", \"set_point\", etc"
          "  - indices: a comma-separated list of indices or \"all\"\n"
          "  - value: the new parameter(s) value (number)\n"
          "e.g. (use double quotation) update_pid_param [''LO'',''kp'',''all'', 0.0] or update_pid_param [''HO'',''kp'',''1,3,5,42'',0.1] or ",
          "args"_arg);

    //e.g: dg ["LO","ki",0.1] 
    m.def("dg", dg,
        "Increment all elements of kp, ki, or kd of the specified controller by a given increment.\n"
        "Usage: increment_pid_param [\"LO\" or \"HO\", \"kp\" or \"ki\" or \"kd\", increment]",
        "args"_arg);



    m.def(
    "build_probe_args_for_method",
    [](std::string method) -> nlohmann::json {
        return build_probe_args_for_method(method);   // returns the args payload
    },
    R"(Return the probe_interaction_data argument for a named method.
    Usage: build_probe_args_for_method "method_1")"
    );

    m.def(
    "probe_interaction_data", probe_interaction_data,
    R"(Apply modal probes and capture telemetry.
    Args: [ [[idx,amp],...], "basis", "OL|CL", "field", N, deadtime_ms? ]
    or:   {"aberrations":[...],"basis":"...","state":"OL|CL","field":"...","N":N,"deadtime_ms":ms}
    Example: probe_interaction_data [[[5,0.05],[12,-0.03]],"zernike","OL","signal",2,50])"
    );

    /// This is where users can really define useful sequence of 
    // offsets / aberrations to apply in closed or open loop! 
    m.def(
        "run_probe_method",
        run_probe_method,
        R"(Run a named probe recipe and save to FITS.
    Usage: probe_method ["method_1", "/optional/dir/or/filename.fits"]
        probe_method {"method":"method_1","path":"/optional/dir/or/filename.fits"}
    Default path = telem_save_path with timestamped file, /usr/local/var/baldr/probes/probe_<method>_<YYYYMMDD_HHMMSS>.fits.)"
    );

    m.def(
        "tt_inj",
        tt_inj,
        R"(Configure Tip/Tilt (LO) injection for system ID.
        Args (array): ["chirp|prbs|sine", amplitude?, "setpoint|command"?]
        Args (object): {"method":"prbs","amplitude":0.02,"apply_to":"setpoint"}
        Returns: {"ok":true, "method":..., "amplitude":..., "apply_to":"setpoint|command", "branch":"LO"})",
            "args"_arg
    );


 }




std::map<std::string, std::string> parse_named_args(int argc, char* argv[]) {
    std::map<std::string, std::string> opts;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            std::cerr << "[ERROR] Unexpected argument format: " << key << std::endl;
            continue;
        }

        key = key.substr(2);  // remove '--'
        if ((i + 1 < argc) && std::string(argv[i+1]).rfind("--", 0) != 0) {
            opts[key] = argv[++i];
        } else {
            opts[key] = "1";  // e.g., for --verbose
        }
    }
    return opts;
}


int main(int argc, char* argv[]) {
    auto args = parse_named_args(argc, argv);

    // Required
    if (!args.count("beam") || !args.count("mask")) {
        std::cerr << "Usage: " << argv[0]
                  << " --beam <id> --mask <name> [--mode bright|faint] [--config file.toml] [--socket tcp://*:6662]"
                  << std::endl;
        return 1;
    }

    int beam_id = std::stoi(args["beam"]);
    std::string phasemask = args["mask"];
    std::string observing_mode = args.count("mode") ? args["mode"] : "bright";

    if (observing_mode != "bright" && observing_mode != "faint") {
        std::cerr << "[ERROR] Invalid mode. Use 'bright' or 'faint'." << std::endl;
        return 1;
    }

    std::string filename;
    if (args.count("config")) {
        filename = args["config"];
        std::cout << "[INFO] Using user config file: " << filename << std::endl;
    } else {
        filename = find_latest_config_file(beam_id, observing_mode);
        std::cout << "[INFO] Using latest config: " << filename << std::endl;
    }
    // Load and parse TOML config
    try {
        config = toml::parse_file(filename);
        std::cout << "Loaded configuration for beam " << beam_id << " from " << filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing file " << filename << ": " << e.what() << std::endl;
        return 1;
    }
    
    // Now set up rtc_config
    std::string beamKey = "beam" + std::to_string(beam_id);
    
    
    try {
        rtc_config = readBDRConfig(config, beamKey, phasemask);


        std::cout << "[INFO] RTC configuration initialized for beam " << beam_id << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing RTC config: " << e.what() << std::endl;
        return 1;
    }


    // set up some DM basis cache 
    dm::load_from_fits_file(kDMBaseFITS);
    std::cout << "Listing available DM basis: " << std::endl;
    list_loaded_bases();
    

    // Open shared memory
    if (!rtc_config.state.simulation_mode) {
        std::cout << "[INFO] Opening real SHM (not simulation)..." << std::endl;

        std::cout << "here in NOT simulation mode" << std::endl;

        // real hardware servers
        cam_host_str = "tcp://192.168.100.2:6667";
        mds_host_str = "tcp://192.168.100.2:5555";


    } else {
        //std::cerr << "[ERROR] Simulation mode not implemented yet." << std::endl;
        //throw std::runtime_error("Simulation mode not implemented");

        std::cout << "here in simulation mode" << std::endl;

        // Use the Fake servers!
        cam_host_str = "tcp://127.0.0.1:6667";
        mds_host_str = "tcp://127.0.0.1:5555";

    }

    // after we set the corrrect host addresses based on simulation mode (need to read in toml first)
    // then we init derived parameters (this requires zmq communication to camera server  - hence why we do it last)
    try {
        rtc_config.initDerivedParameters();
        std::cout << "[INFO] RTC configuration initialized for beam " << beam_id << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing RTC config: " << e.what() << std::endl;
        return 1;
    }
    //rtc_config.initDerivedParameters();


    ImageStreamIO_openIm(&subarray, ("baldr" + std::to_string(beam_id)).c_str());

    std::string name = "dm" + std::to_string(beam_id) + "disp02";
    std::string name0 = "dm" + std::to_string(beam_id);

    if (ImageStreamIO_openIm(&dm_rtc, name.c_str()) != IMAGESTREAMIO_SUCCESS) {
        std::cerr << "[ERROR] Failed to open DM SHM: " << name << std::endl;
        return 1;
    }
    if (ImageStreamIO_openIm(&dm_rtc0, name0.c_str()) != IMAGESTREAMIO_SUCCESS) {
        std::cerr << "[ERROR] Failed to open DM master SHM: " << name0 << std::endl;
        return 1;
    }


    int dm_cmd_length = dm_rtc.md ? static_cast<int>(dm_rtc.md->nelement) : 0;
    init_openloop_offsets(dm_cmd_length);   // publishes zero offsets of correct length
    std::cout << "DM command length in shm metadata = " << dm_cmd_length << std::endl;

    // Start RTC threads
    std::thread rtc_thread(rtc);
    std::thread telemetry_thread(telemetry);

    std::cout << "[INFO] RTC and telemetry threads started." << std::endl;

    
    // prepare commander arguments
    std::vector<std::string> commander_args = {argv[0]};

    if (args.count("socket")){
        commander_args.push_back("--socket");
        commander_args.push_back(args["socket"]);
    }

    // this is a safe C style allocation of strings so if commander_args arn't allive we dont get invalid pointer
    std::vector<std::unique_ptr<char[]>> argv_storage;
    std::vector<char*> commander_argv;
    for (const auto& s : commander_args){
        argv_storage.emplace_back(std::make_unique<char[]>(s.size()+1));
        std::strcpy(argv_storage.back().get(), s.c_str());
        commander_argv.push_back(argv_storage.back().get());
    }
    
    // this was the old functional but maybe less robust way fo commander_argv
    //std::vector<char*> commander_argv;
    //for (auto& s : commander_args )
    //    commander_argv.push_back(&s[0]);
    
    int commander_argc = commander_argv.size();
    // Start commander server
    //commander::Server s(argc, argv);
    commander::Server s(commander_argc, commander_argv.data());

    // test we past commander input 
    std::cout << "commander server started" << std::endl;

    s.run();

    // Clean shutdown
    servo_mode = SERVO_STOP;
    rtc_thread.join();
    telemetry_thread.join();

    std::cout << "DONE" << std::endl;
    return 0;
}


// // previous working veruis 
// int main(int argc, char* argv[]) {
//     if (argc < 3 || argc > 5) {
//         std::cerr << "Usage: " << argv[0] << " <beam_id> <phaseMask> [mode (bright|faint)] [optional config_filename.toml]" << std::endl;
//         return 1;
//     }

//     // Mandatory inputs
//     beam_id = std::stoi(argv[1]);
//     phasemask = argv[2];

//     // Optional mode
//     if (argc >= 4) {
//         observing_mode = argv[3];
//         if (observing_mode != "bright" && observing_mode != "faint") {
//             std::cerr << "[ERROR] Invalid observing mode. Must be 'bright' or 'faint'." << std::endl;
//             return 1;
//         }
//     }

//     // Optional user configuration file
//     if (argc == 5) {
//         user_config_filename = argv[4];
//         user_provided_config = true;
//     }

//     std::string filename;
//     if (user_provided_config) {
//         filename = user_config_filename;
//         std::cout << "[INFO] Using user-provided config file: " << filename << std::endl;
//     } else {
//         try {
//             filename = find_latest_config_file(beam_id, observing_mode); // <--- updated find_latest_config_file
//             std::cout << "[INFO] Using latest config for mode '" << observing_mode << "': " << filename << std::endl;
//         } catch (const std::exception& e) {
//             std::cerr << "Error finding latest config file: " << e.what() << std::endl;
//             return 1;
//         }
//     }

//     // Load and parse TOML config
//     try {
//         config = toml::parse_file(filename);
//         std::cout << "Loaded configuration for beam " << beam_id << " from " << filename << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "Error parsing file " << filename << ": " << e.what() << std::endl;
//         return 1;
//     }

//     // Now set up rtc_config
//     std::string beamKey = "beam" + std::to_string(beam_id);
//     try {
//         rtc_config = readBDRConfig(config, beamKey, phasemask);
//         rtc_config.initDerivedParameters();
//         std::cout << "[INFO] RTC configuration initialized for beam " << beam_id << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "Error initializing RTC config: " << e.what() << std::endl;
//         return 1;
//     }

//     // Open shared memory
//     if (!rtc_config.state.simulation_mode) {
//         std::cout << "[INFO] Opening real SHM (not simulation)..." << std::endl;

//         ImageStreamIO_openIm(&subarray, ("baldr" + std::to_string(beam_id)).c_str());

//         std::string name = "dm" + std::to_string(beam_id) + "disp02";
//         std::string name0 = "dm" + std::to_string(beam_id);

//         if (ImageStreamIO_openIm(&dm_rtc, name.c_str()) != IMAGESTREAMIO_SUCCESS) {
//             std::cerr << "[ERROR] Failed to open DM SHM: " << name << std::endl;
//             return 1;
//         }
//         if (ImageStreamIO_openIm(&dm_rtc0, name0.c_str()) != IMAGESTREAMIO_SUCCESS) {
//             std::cerr << "[ERROR] Failed to open DM master SHM: " << name0 << std::endl;
//             return 1;
//         }

//     } else {
//         std::cerr << "[ERROR] Simulation mode not implemented yet." << std::endl;
//         throw std::runtime_error("Simulation mode not implemented");
//     }

//     // Start RTC threads
//     std::thread rtc_thread(rtc);
//     std::thread telemetry_thread(telemetry);

//     std::cout << "[INFO] RTC and telemetry threads started." << std::endl;

//     // Start commander server
//     commander::Server s(argc, argv);
//     s.run();

//     // Clean shutdown
//     servo_mode = SERVO_STOP;
//     rtc_thread.join();
//     telemetry_thread.join();

//     std::cout << "DONE" << std::endl;
//     return 0;
// }





// before we introduced faint mode 
// // configure like ./baldr 1 H3 
// int main(int argc, char* argv[]) {
//     // We expect arguments: <beam_id> <phaseMask>
//     if (argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <beam_id1> <phaseMask1>" << std::endl;
//         return 1;
//     }
//     // if (argc != 3 && argc != 4) {
//     //     std::cerr << "Usage: " << argv[0] << " <beam_id1> <phaseMask1> [optional loop_time (us)]" << std::endl;
//     //     return 1;
//     // }

//     beam_id = std::stoi(argv[1]);
//     phasemask = argv[2];


//     // if (argc == 4) {
//     //     loop_time = std::stof(argv[3]);
//     //     loop_time_override = true;
//     //     if (loop_time <= 0.0f) {
//     //         std::cerr << "[ERROR] Invalid loop_time specified. Must be > 0." << std::endl;
//     //         return 1;
//     //     }
//     //     std::cout << "[INFO] User override loop_time = " << loop_time << " seconds (" << 1.0f / loop_time << " Hz)" << std::endl;
//     // }


//     // Compute the configuration filename and parse the TOML file.
//     //std::string filename = "/usr/local/etc/baldr/baldr_config_" + std::to_string(beam_id) + ".toml";

//     std::string filename;
//     try {
//         filename = find_latest_config_file(beam_id);
//         std::cout << "Using latest config: " << filename << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "Error finding latest config file: " << e.what() << std::endl;
//         exit(1);
//     }

//     try {
//         config = toml::parse_file(filename);
//         std::cout << "Loaded configuration for beam " << beam_id << " from " << filename << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "Error parsing file " << filename << ": " << e.what() << std::endl;
//         return 1;
//     }

//     //to create a global list of bdr_rtc_config structs:
//     std::string beamKey = "beam" + std::to_string(beam_id);
//     try {
//         rtc_config = readBDRConfig(config, beamKey, phasemask);
//         std::cout << "after read in baldr main rtc.ctrl_LO_config.kp.size() = " << rtc_config.ctrl_LO_config.kp.size() << std::endl;
//         rtc_config.initDerivedParameters();
//         std::cout << "finished init of run_time parameters" << std::endl;
//         std::cout << "after initDerived parameters in baldr main rtc.ctrl_LO_config.kp.size() = " << rtc_config.ctrl_LO_config.kp.size() << std::endl;

//         std::cout << "Initialized configuration for beam " << beam_id << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "Error initializing configuration for beam " << beam_id << ": " << e.what() << std::endl;
//         return 1;
//     }

//     // if user input for loop time then  - we should 
//     // if (!loop_time_override) {
//     //     loop_time = 1000000.0f / rtc_config.fps; // us seconds
//     //     std::cout << "[INFO] Default loop_time set from RTC config FPS: loop_time = " 
//     //               << loop_time << " seconds (" << rtc_config.fps << " Hz)" << std::endl;
//     // }


//     // The C-red image for the baldr subarray is /dev/shm/baldrN.im.shm, 
//     // where N is the beam number.
//     // 
//     //     ImageStreamIO_openIm(&subarray, ("baldr" + std::to_string(beam_id) ).c_str());
        
//     //     // Open the DM image. It is e.g. /dev/shm/dm2disp02.im.shm for beam 2.
//     //     std::string dm_filename = "dm" + std::to_string(beam_id) + "disp02" ;
//     //     ImageStreamIO_openIm(&dm_rtc, dm_filename.c_str());
//     // 
//     if (!rtc_config.state.simulation_mode) {
//         std::cout << "SHM configuration not in simulation mode" << std::endl;
//         // Real mode: open real images from the shared memory.
//         ImageStreamIO_openIm(&subarray, ("baldr" + std::to_string(beam_id)).c_str());

//         // Form the shared-memory name, e.g. "dm2disp02"
//         std::string name = "dm" + std::to_string(beam_id) + "disp02";
//         std::string name0 = "dm" + std::to_string(beam_id);
//         std::cout << "[INFO] Opening DM SHM \"" << name << "\"\n";
        
//         if (ImageStreamIO_openIm(&dm_rtc, name.c_str()) != IMAGESTREAMIO_SUCCESS) {
//             std::cerr << "[ERROR] Failed to open DM SHM \"" << name << "\"\n";
//             return 1;
//         }
//         if (ImageStreamIO_openIm(&dm_rtc0, name0.c_str()) != IMAGESTREAMIO_SUCCESS) {
//             std::cerr << "[ERROR] Failed to open DM SHM \"" << name0 << "\"\n";
//             return 1;
//         }
//         // BCB        
//         // // Open the DM image. (For beam 2, for example, this may be: "/dev/shm/dm2disp02.im.shm")
//         // std::string dm_filename = "dm" + std::to_string(beam_id) + "disp02";
//         // ImageStreamIO_openIm(&dm_rtc, dm_filename.c_str());

//         // // master process for DM 
//         // std::string name0 = "dm" + std::to_string(beam_id);
//         // ImageStreamIO_openIm(&dm_rtc0, name0.c_str());

//         // //add.
//         // ImageStreamIO_semflush(&dm_rtc0, /*index=*/-1);

//     } else {
//         // Simulation mode: if not yet implemented, raise an error.
//         std::cerr << "Simulation mode not implemented. Aborting." << std::endl;
//         throw std::runtime_error("Simulation mode not implemented");
        
//         // Alternatively, if you prefer to use dummy simulation images,
//         // you might do something like:
//         // ImageStreamIO_openIm(&subarray, "simulated_baldr.im.shm");
//         // ImageStreamIO_openIm(&dm_rtc, "simulated_dm.im.shm");
//     }
//     // Start the main RTC and telemetry threads. 


//     std::thread rtc_thread(rtc);
//     std::thread telemetry_thread(telemetry);

//     std::cout << "after rtc thread started in  main rtc.ctrl_LO_config.kp.size() = " << rtc_config.ctrl_LO_config.kp.size() << std::endl;
//     // Initialize the commander server and run it
//     commander::Server s(argc, argv);
//     s.run();

//     // Join the  thread
//     servo_mode = SERVO_STOP;
//     rtc_thread.join();
//     telemetry_thread.join();

//     // Continue with your RTC main loop using rtc_config_list...
//     std::cout << "DONE" << std::endl;
    
//     return 0;
// }

