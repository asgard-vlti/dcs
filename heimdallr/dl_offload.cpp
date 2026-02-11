#include "heimdallr.h"
#include <nlohmann/json.hpp>
#include <fstream>
// usleep for the controllino - this is actually a wait because we are 
// waiting for an external device.
#define CONTROLLINO_USLEEP 1000
#define HFO_DEADBAND 3.5 //Deadband of HFO motion in microns of OPD. This is a sum over all DLs

// Local globals.
sem_t sem_offload;
int controllinoSocket;
Eigen::Vector4d next_offload;
// This is non-zero to make sure if using the Piezos the DLs are centred.
// It has a bad name - it is actually the sum of the next_offload
// and the search_offset, i.e. it is a total offload.
Eigen::Vector4d last_offload = Eigen::Vector4d::Constant(0.01);
int search_ix = 0;
int search_length = 0;
int search_dl = 0;
double search_delta = 0.5;
double search_start = 0.0;
uint search_dt_ms = 200; // Time between search steps in ms
double search_snr_threshold = 10.0; // SNR threshold to stop searching

double hfo_offsets[N_TEL] = {0.0, 0.0, 0.0, 0.0};
auto last_hfo = std::chrono::high_resolution_clock::now();
// Add a local global to track last HFO offset send time
auto last_hfo_offset = std::chrono::high_resolution_clock::now();

// Initialize global ZMQ variables for MultiDeviceServer
zmq::context_t mds_zmq_context(1);
int timeout_ms = 1000;
zmq::socket_t mds_zmq_socket(mds_zmq_context, zmq::socket_type::req);
const std::string mds_host_str = "tcp://192.168.100.2:5555";
zmq::context_t wag_rmn_context(1);
zmq::socket_t wag_rmn_socket(wag_rmn_context, zmq::socket_type::req);
const std::string wag_rmn_host_str = "tcp://192.168.100.1:7050";
bool mds_zmq_initialized = false, controllino_initialized = false, wag_rmn_initialized = false;

// Initialize the connection to wag for the RMN relay
void init_wag_rmn() {
    if (!wag_rmn_initialized) {
        wag_rmn_socket.setsockopt(ZMQ_CONNECT_TIMEOUT, &timeout_ms, sizeof(timeout_ms));
        wag_rmn_socket.setsockopt(ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
        // the next line will throw an exception if it fails, in which case
        // RMN isn't initialized. We need to catch this exception.
        try {
            wag_rmn_socket.connect(wag_rmn_host_str);
            wag_rmn_initialized = true;
        } catch (const zmq::error_t& e) {
            wag_rmn_initialized = false;
            std::cerr << "Error initializing WAG RMN: " << e.what() << std::endl;
        }
    }
}

// Initialize the MDS ZMQ connection, for the HFO connection.
void init_mds_zmq() {
    if (!mds_zmq_initialized) {
        mds_zmq_socket.setsockopt(ZMQ_CONNECT_TIMEOUT, &timeout_ms, sizeof(timeout_ms));
        mds_zmq_socket.setsockopt(ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms));
        mds_zmq_socket.connect(mds_host_str);
        mds_zmq_initialized = true;
    }
}

// Initialize the Controllino connection, for the piezo connection.
void init_controllino() {
    if (controllino_initialized) return;
    // Connect to the Controllino
    controllinoSocket = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(23);
    serverAddress.sin_addr.s_addr = inet_addr("192.168.100.10");
    connect(controllinoSocket, (struct sockaddr*)&serverAddress,
            sizeof(serverAddress));

    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    setsockopt(controllinoSocket, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
    controllino_initialized = true;
}

// Send a command to MDS and wait for a reply.
std::string send_mds_cmd(const std::string& message) {
#ifdef SIMULATE
    return "0.0";
#endif
    init_mds_zmq();
    mds_zmq_socket.send(zmq::buffer(message), zmq::send_flags::none);
    zmq::message_t reply;
    auto result = mds_zmq_socket.recv(reply, zmq::recv_flags::none);
    if (!result.has_value()) {
        fmt::print("Timeout or error receiving reply from MDS.");
        mds_zmq_initialized=false;
        return "0.0";
    }
    return std::string(static_cast<char*>(reply.data()), reply.size());
}

// Set the delay line offsets (from the servo loop). Units are microns of OPD.
void set_delay_lines(Eigen::Vector4d dl) {
    next_offload=dl;
}

void add_to_delay_lines(Eigen::Vector4d dl) {
    // First, decide if we are ignoring this command
    if (delay_line_type =="hfo"){
        // Only send if more than 0.5 since last offset
        auto now = std::chrono::high_resolution_clock::now();
        double seconds_since_last = std::chrono::duration<double>(now - last_hfo_offset).count();
        if (seconds_since_last < 0.5) {
            fmt::print("Not enough time for a new offload.\n");
            return;
        }
        // Check to find the total offload requested, adding all values of
        double total_offload = 0.0;
        for (int i = 0; i < N_TEL; i++) {
            total_offload += std::fabs(dl(i));
        }
        // If the total offload is less than the deadband of about  micron, do not send
        if (total_offload < HFO_DEADBAND) return;
    }
    next_offload += dl;
}

// Set one delay line value. 
void set_delay_line(int dl, double value) {
    // This function sets the delay line value for a specific telescope.
    // The value is in K1 wavelengths.
    if (dl == 0 || dl >= N_TEL) {
        std::cout << "Delay line number out of range" << std::endl;
        return;
    }
    next_offload[dl-1] = value;
}

void start_search(uint search_dl_in, double start, double stop, double rate, uint dt_ms, double threshold) {
    // This function sets the search parameters for the delay line.
    search_ix = 0;
    search_length = (int)((stop - start) / rate);
    search_dl = search_dl_in;
    search_delta = rate;
    search_start = start;
    search_dt_ms = dt_ms;
    search_snr_threshold = threshold;
}

void move_piezos(){
#ifdef SIMULATE
    return;
#endif
    // This function sets the piezo delay line to the stored value.
    char message[20];
    char buffer[64] = { 0 };
    int dl_value;
    init_controllino();
    // This next loop should be turned into a single function for rapid movement.
    for (int i = 0; i < N_TEL; i++) {
        if (last_offload(i) != next_offload(i) + search_offset(i)){
            dl_value = 2048 + (int)( (next_offload(i) + search_offset(i)) / OPD_PER_PIEZO_UNIT);
            sprintf(message, "a%d %d\n", i, dl_value);
            recv(controllinoSocket, buffer, sizeof(buffer), 0);
            if (strlen(buffer) > 0) std::cout << "Before starting, controllino: " << buffer << std::endl;
            send(controllinoSocket, message, strlen(message), 0);
            std::cout << "Sending to controllino: " << message << std::endl;
            usleep(CONTROLLINO_USLEEP);
            recv(controllinoSocket, buffer, sizeof(buffer), 0);
            if (buffer[0] != 'S') {
                std::cout << "Controllino error! Setting state uninitialised." << std::endl;
                controllino_initialized = false;
                return;
            }
        }
    }
    last_offload = next_offload + search_offset;
}

// This function sets the HFO actuators to the stored value.
void move_hfo(){
    for (int i = 0; i < N_TEL; i++) {
        if ( last_offload(i)  != next_offload(i) + search_offset(i) ) {
            // Set the delay line value for the current telescope (value in mm of physical motion)
            double dl_value = hfo_offsets[i] - (next_offload(i) + search_offset(i)) * 0.0005;
            std::string message = "moveabs HFO" + std::to_string(i+1) + " " + std::to_string(dl_value);
            fmt::print(message + "\n");
            fmt::print(send_mds_cmd(message));
            last_offload(i) = next_offload(i) + search_offset(i);
        } 
    }
    last_hfo_offset = std::chrono::high_resolution_clock::now();
}

/* Example of the RMN relay command to wag
{"command" :
{ "name" : "writermn",
  "time" : "2024-08-27T12:12:31",
  "parameter" :
[
{"name" : "opd_offset",
 "value" : [1.1, 2.2, 3.3, 4.4]},
{"name" : "offset_valid",
 "value" : [1, 1, 0, 1]},
{"name" : "fringe_detect",
 "value" : [1, 0, 0, 1]}
]}

A good reply string is:
{
	"reply" :
	{
		"time" : "2025-09-09T20:28:33",
		"content" : "OK"
	}
}
*/
void move_main_dl()
{
    // Only execute if wag_rmn is initialized.
    // (!!! following the method of other functions, we could initialize 
    // here, but that risks repeated failed attempts)
    if (!wag_rmn_initialized) return;
    //fmt::print("Here...\n");
    // Build the JSON message
    nlohmann::json j;
    j["command"]["name"] = "writermn";
    // Use current time as ISO string
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&now_c), "%Y-%m-%dT%H:%M:%S");
    j["command"]["time"] = ss.str();

    // Fill parameters
    nlohmann::json params = nlohmann::json::array();
    params.push_back({{"name", "opd_offset"}, {"value", {-next_offload(0), -next_offload(1), -next_offload(2), -next_offload(3)}}});
    // Example: offset_valid and fringe_detect can be filled with dummy or real values as needed
    if ((offload_mode == OFFLOAD_OFF))
    	params.push_back({{"name", "offset_valid"}, {"value", {0, 0, 0, 0}}});
    else
    	params.push_back({{"name", "offset_valid"}, {"value", {1, 1, 1, 1}}});
    if (control_u.fringe_found)
      params.push_back({{"name", "fringe_det"}, {"value", {1, 1, 1, 1}}});
    else
      params.push_back({{"name", "fringe_det"}, {"value", {0, 0, 0, 0}}});
    j["command"]["parameter"] = params;

    std::string msg = j.dump(); // No newlines
    //fmt::print("Sent to wag: {} \n", j.dump());

    wag_rmn_socket.send(zmq::buffer(msg), zmq::send_flags::none);
    zmq::message_t reply;
    auto result = wag_rmn_socket.recv(reply, zmq::recv_flags::none);
    if (result.has_value()) {
        //std::string reply_str(static_cast<char*>(reply.data()), reply.size());
        //fmt::print("WAG RMN reply: {}\n", reply_str);
    } else {
        wag_rmn_initialized=false;
        fmt::print("Timeout or error receiving reply from WAG RMN.\n");
    }
    last_offload = next_offload + search_offset;
}

// The main thread function
int last_offload_mode=-1;
void dl_offload(){
    // Initialize semaphore for offload timing
    sem_init(&sem_offload, 0, 0);

    // Try to connect just once to WAG RMN.
    init_wag_rmn();
    // Connect to MDS and find the delay line positions.
    for (int i = 0; i < N_TEL; i++) {
        std::string message = "read HFO" + std::to_string(i+1);
        std::string reply = send_mds_cmd(message);
        fmt::print(reply);
        hfo_offsets[i] = std::stod(reply);
        fmt::print("HFO{} offset: {}\n", i+1, hfo_offsets[i]);
    }
    set_delay_lines(Eigen::Vector4d::Zero());

    auto last_search_time = std::chrono::high_resolution_clock::now();

    while (keep_offloading) {
        // Wait for the next offload - nominally 200Hz max, but controlled
        // by the fringe tracker thread.
        sem_wait(&sem_offload);

        // Check if we need to zero the dl_offload
        if (zero_offload) {
            if (delay_line_type == "hfo") {
                // Read the current HFO positions and set the offsets to these values
                for (int i = 0; i < N_TEL; i++) {
                    std::string message = "read HFO" + std::to_string(i+1);
                    std::string reply = send_mds_cmd(message);
                    fmt::print(reply);
                    hfo_offsets[i] = std::stod(reply);
                    fmt::print("HFO{} offset: {}\n", i+1, hfo_offsets[i]);
                    last_offload(i) = 0.0;
                    next_offload(i) = 0.0;
                }
            }
            zero_offload=false;
        }

        auto now = std::chrono::high_resolution_clock::now();
        // Only run linear search block if enough time has passed
        if (search_ix < search_length) {
            double ms_since_last = std::chrono::duration<double, std::milli>(now - last_search_time).count();
            if (ms_since_last >= search_dt_ms) {
                last_search_time = now;
                double max_snr = 0.0;
                // Check all baselines associated with the current delay line,
                // using beam_baselines[search_dl]
                for (int i = 0; i < N_TEL-1; i++) {
                    // Get the SNR for the current baseline
                    double snr = baselines.pd_snr(beam_baselines[search_dl][i]);
                    if (snr > max_snr) {
                        max_snr = snr;
                    }
                }
                fmt::print("Search beam max SNR: {}\n", max_snr);
                // Check if the SNR is above the threshold
                if (max_snr > search_snr_threshold) {
                    search_length = 0;
                } else {
                    // Move the piezo to the next position
                    next_offload(search_dl) = search_start + search_ix * search_delta;
                    // Indicate that there is another piezo offload to do.
                    search_ix++;
                }
            }
        } 
        // Has the offload changed? If so, consider doing it and log to file.
        bool offload_changed=false;
        for (int i=0;i<N_TEL;i++)
    	  if (last_offload(i) != next_offload(i) + search_offset(i))
    		offload_changed=true;
    	if ((offload_changed) || (offload_mode != last_offload_mode)){
		    // Do the offload! It is up to the specific function to check if the offload
		    // is significant enough to do - only if so, it will move and 
		    // update last_offload.
		    if (delay_line_type == "piezo") {
		        // Move the piezo delay line to the next position
		        move_piezos();
		    } else if (delay_line_type == "hfo") {
		        // Move the delay line to the next position
		        move_hfo();
		    } else if (delay_line_type == "rmn") {
		        move_main_dl();
		    } else {
		        std::cout << "Delay line type not recognised" << std::endl;
		    }
		    // Log delay line type and values to file with timestamp
		    std::ofstream log_file("/data/dl_offload.log", std::ios::app);
		    auto lnow = std::chrono::system_clock::now();
		    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(lnow.time_since_epoch()).count();
		    double timestamp = ms / 1000.0;
		    log_file << fmt::format("{:.3f} {} {:.6f} {:.6f} {:.6f} {:.6f}\n",
		        timestamp,
		        delay_line_type,
		        next_offload(0), next_offload(1), next_offload(2), next_offload(3));  
		 }      
    }
    close(controllinoSocket);
}

