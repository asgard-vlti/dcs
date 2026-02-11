#define TOML_IMPLEMENTATION
#include "heimdallr.h"
#include <commander/commander.h>
#include <math.h>
#include <unistd.h>
// Commander struct definitions for json. This is in a separate file to keep the main code clean.
#include "commander_structs.h"
using namespace std::complex_literals;
extern "C" {
#include <b64/cencode.h> // Base64 encoding, in C so Frantz can see how it works.
}
//----------Globals-------------------
// The input configuration
toml::table config;

// Servo parameters. These are the parameters that will be adjusted by the commander
int servo_mode=SERVO_OFF;
int offload_mode=OFFLOAD_OFF;
uint offload_time_ms=10;
PIDSettings pid_settings;
ControlU control_u;
ControlA control_a;
Baselines baselines;
Bispectrum bispectra_K1[N_CP];
Bispectrum bispectra_K2[N_CP];

// Generally, we either work with beams or baselines, so have a separate lock for each.
std::mutex baseline_mutex, beam_mutex;

std::atomic<bool> zero_offload=false; // Atomic variable to zero the dl_offload

//The forward Fourier transforms
ForwardFt *K1ft, *K2ft;

// Offload globals
bool keep_offloading = true;
Eigen::Vector4d search_offset = Eigen::Vector4d::Zero();
std::string delay_line_type="rmn";

IMAGE DMs[N_TEL];
IMAGE master_DMs[N_TEL];

// Utility functions

// Based on https://sourceforge.net/p/libb64/git/ci/master/tree/examples/c-example1.c
// If bandwith is an issue, we could compress the data before encoding it.
std::string encode(const char* input, unsigned int size)
{
	/* set up a destination buffer large enough to hold the encoded data */
    // print the size of the input
    //std::cout << "Size of input: " << size << std::endl;
	//char* output = (char*)malloc(size*4/3 + 4); /* large enough */
	char* output = (char*)malloc(size*2); /* large enough */
	/* keep track of our encoded position */
	char* c = output;
	/* store the number of bytes encoded by a single call */
	int cnt = 0;
	/* we need an encoder state */
	base64_encodestate s;
	
	/*---------- START ENCODING ----------*/
	/* initialise the encoder state */
	base64_init_encodestate(&s);
	/* gather data from the input and send it to the output */
	cnt = base64_encode_block(input, size, c, &s);
	c += cnt;
	/* since we have encoded the entire input string, we know that 
	   there is no more input data; finalise the encoding */
	cnt = base64_encode_blockend(c, &s);
	c += cnt;
	/*---------- STOP ENCODING  ----------*/
	
    /* we want to convert to a C++ string, so null-terminate */
	*c = 0;
    // Convert the char* to a string
    std::string output_str(output);

    // Free the memory
    free(output);

    return output_str;
}

//----------commander functions from here---------------
void linear_search(uint beam, double start, double stop, double rate, uint search_dt_ms, double search_snr_threashold) {
    if ((beam > N_TEL) || (beam == 0)) {
        std::cout << "Beam number (arg 0) out of range (1 to " << N_TEL-1 << ")" << std::endl;
        return;
    }
    beam_mutex.lock();
    
    // Set the delay line to the start position
    set_delay_line(beam, start);

    // Set the piston DM to zero. !!! Can't be called from this thread.
    //set_dm_piston(Eigen::Vector4d::Zero());

    beam_mutex.unlock();
    usleep(DELAY_MOVE_USEC); // Wait for the delay line to move
    // Set the SNR values to zero.
    baseline_mutex.lock();
    baselines.gd_snr = Eigen::VectorXd::Zero(N_BL); 
    baselines.pd_snr = Eigen::VectorXd::Zero(N_BL);
    baseline_mutex.unlock();

    // Start the search.
    start_search(beam, start,  stop, rate, search_dt_ms, search_snr_threashold);
    fmt::print("Starting search for beam {} from {} to {} at rate {}\n", 
        beam, start, stop, rate);
    return;
}

// Set the servo mode
void set_servo_mode(std::string mode) {
    if (mode == "off") {
        servo_mode = SERVO_OFF;
    } else if (mode == "simple") {
        servo_mode = SERVO_SIMPLE;
    } else if (mode == "fight") {
        servo_mode = SERVO_FIGHT;
    } else if (mode == "lacour") {
        servo_mode = SERVO_LACOUR;
    } else if (mode == "on") {
        // "on" means lacour with nested offload
        servo_mode = SERVO_LACOUR;
        control_u.dl_offload.setZero();
        offload_mode = OFFLOAD_NESTED;
    } else {
        std::cout << "Servo mode not recognised" << std::endl;
        return;
    }
    // Reset the control_u parameters
    control_u.dl.setZero();
    control_u.piezo.setZero();
    control_u.dm_piston.setZero();
    control_u.search_Nsteps=0;
    std::cout << "Servo mode updated to " << servo_mode << std::endl;
    return;
}

// Set the offload time
void set_offload_time(uint time) {
    if (time < 10 || time > 10000) {
        std::cout << "Offload time out of range (0.01 to 10s)" << std::endl;
        return;
    }
    offload_time_ms = time;
    std::cout << "Offload time updated to " << offload_time_ms << " ms" << std::endl;
    return;
}

// Set the offload mode
std::string set_offload_mode(std::string mode) {
    if (mode == "off") {
        offload_mode = OFFLOAD_OFF;
    } else if ((mode == "nested") || (mode == "nest")) {
        offload_mode = OFFLOAD_NESTED;
        // Reset the offload to zero.
        control_u.dl_offload.setZero();
    } else if (mode == "gd") {
        offload_mode = OFFLOAD_GD;
        servo_mode = SERVO_OFF;
    } else if ((mode == "man") || (mode =="manual")) {
        offload_mode = OFFLOAD_MANUAL;
    } else {
        return "ERROR: Offload mode not recognised";
    }
    control_u.search_Nsteps=0;
    return "OK";
}

// Set the delay line offsets (from the servo loop)
void set_search_offset(std::vector<double> offset_in_microns) {
    for (uint i = 0; i < N_TEL; i++) {
        if (i < offset_in_microns.size()) {
            search_offset(i) = offset_in_microns[i];
        } else {
            search_offset(i) = 0.0;
        }
    }
    std::cout << "Search offset updated to " << search_offset.transpose() << std::endl;
}

// Get the delay line offsets (from the servo loop)
std::vector<double> get_search_offset(void) {
    std::vector<double> offsets(N_TEL);
    for (uint i = 0; i < N_TEL; i++) {
        offsets[i] = search_offset(i);
    }
    return offsets;
}

// EncodedImage  // std::string
EncodedImage get_ps(std::string filter) {
    // We get the power spectrum for one filter. This is a 2D array.
    // !!!! Warning: This is not thread-safe - we could have a mutex
    // for each filter, but that would be overkill for now.

    ForwardFt *ft;
    bool instantaneous=false;
    std::string encoded_ps;
    if (filter == "K1") {
        ft = K1ft;
    } else if (filter == "K2") {
        ft = K2ft;
    } else if (filter == "K1i") {
        ft = K1ft;
        instantaneous=true;
    } else if (filter == "K2i") {
        ft = K2ft;
        instantaneous=true;
    } else {
        throw std::runtime_error("Filter not recognised - please edit this code for a better error response");
    }
    unsigned int sz_in_bytes = ft->subim_sz * (ft->subim_sz / 2 + 1) * sizeof(double);
    // Encoding doesn't over-write the array, so pass directly to encode.
    if (instantaneous){
        encoded_ps = encode((char*)ft->power_spectra[ft->ps_index], sz_in_bytes);
    } else {
        encoded_ps = encode((char*)ft->power_spectrum, sz_in_bytes);
    }
    EncodedImage ei = {ft->subim_sz, ft->subim_sz / 2 + 1, "double", encoded_ps};
    return ei; //encoded_ps;  
}

// Set the phase delay gain.
void set_gain(double gain) {
    pid_settings.mutex.lock();
    pid_settings.kp = gain;
    pid_settings.mutex.unlock();
}

// Set the Group Delay integral gain. This has different meanings for different 
// servo loop types.
void set_ggain(double gain) {
    pid_settings.mutex.lock();
    pid_settings.gd_gain = gain / baselines.n_gd_boxcar;
    pid_settings.mutex.unlock();
}

// Set the offload gain for 'offload "gd"' mode.
void set_offload_gd_gain(double gain) {
    pid_settings.mutex.lock();
    pid_settings.offload_gd_gain = gain;
    pid_settings.mutex.unlock();
}

// Set the delay line type (doesn't have to be the main delay lines via RMN)
void set_delay_line_type(std::string type) {
    static const std::set<std::string> valid_types = {"piezo", "hfo", "rmn"};
    if (valid_types.count(type)) {
        delay_line_type = type; 
        initialize_delay_line(type);
        std::cout << "Delay line type updated to " << delay_line_type << std::endl;
    } else {
        std::cout << "Delay line type not recognised: " << type << std::endl;
    } 
}

// A wrapper for set_delay_lines that takes 4 doubles as input.
std::string set_delay_lines_wrapper(double delay1=0.0, double delay2=0.0, double delay3=0.0, double delay4=0.0) {
    if (offload_mode == OFFLOAD_OFF) return "ERROR: Offloads off. Set to manual to use the dls command.";
    Eigen::Vector4d delays = Eigen::Vector4d::Zero();
    delays(0) = delay1;
    delays(1) = delay2;
    delays(2) = delay3;
    delays(3) = delay4;
    set_delay_lines(delays);
    // Lock the beam mutex
    beam_mutex.lock();
    control_u.search_Nsteps = 0;
    beam_mutex.unlock();
    return "OK";
}


// Add setter functions for thresholds
// !!! ToDo: These should probably be in a struct like the PID settings,
 // with a mutex, to avoid locking the whole baseline mutex when we want to change one threshold.
void set_gd_threshold(double val) { gd_threshold = val; }
void set_pd_threshold(double val) { pd_threshold = val; }
void set_gd_search_reset(double val) { gd_search_reset = val; }

Status get_status() {
    Status status;
    // Get the status of the system. This is a simple struct with the
    // values we want to send back to the commander.
    // We have to initialise everything to zero or we get a core dump!
    // !!! Should also truncate number of decimals.
    status.gd_bl = std::vector<double>(N_BL);
    status.pd_bl = std::vector<double>(N_BL);
    status.gd_tel = std::vector<double>(N_TEL);
    status.pd_tel = std::vector<double>(N_TEL);
    status.gd_snr = std::vector<double>(N_BL);
    status.pd_snr = std::vector<double>(N_BL);
    status.v2_K1 = std::vector<double>(N_BL);
    status.v2_K2 = std::vector<double>(N_BL);
    status.closure_phase_K1 = std::vector<double>(N_CP);
    status.closure_phase_K2 = std::vector<double>(N_CP);
    status.dl_offload = std::vector<double>(N_TEL);
    status.dm_piston = std::vector<double>(N_TEL);
    status.pd_av = std::vector<double>(N_BL);
    status.pd_av_filtered = std::vector<double>(N_BL);
    status.gd_phasor_real = std::vector<double>(N_BL);
    status.gd_phasor_imag = std::vector<double>(N_BL);
    status.test_ix = control_u.test_ix;
    status.test_n = control_u.test_n;
    status.itime = control_u.itime;

    // Now fill these in with the values from the control structures.
    for (int i = 0; i < N_BL; i++) {
        status.gd_bl[i] = std::round(baselines.gd(i)* 1000.0)/1000.0;
        status.pd_bl[i] = std::round(baselines.pd(i)* 1000.0)/1000.0;
        status.gd_snr[i] = std::round(baselines.gd_snr(i)* 100.0)/100.0;
        status.pd_snr[i] = std::round(baselines.pd_snr(i)* 100.0)/100.0;
        status.v2_K1[i] = std::round(baselines.v2_K1(i)* 10000.0)/10000.0;
        status.v2_K2[i] = std::round(baselines.v2_K2(i) * 10000.0)/10000.0;
        status.pd_av[i] = std::round(baselines.pd_av(i)* 1000.0)/1000.0; //Not needed anymore !!!
        status.pd_av_filtered[i] = std::round(baselines.pd_av_filtered(i)* 1000.0)/1000.0; //Not needed anymore !!!
        status.gd_phasor_real[i] = std::round(std::real(baselines.gd_phasor(i))* 10.0)/10.0;
        status.gd_phasor_imag[i] = std::round(std::imag(baselines.gd_phasor(i))* 10.0)/10.0;
    }
    for (int i = 0; i < N_TEL; i++) {
        status.gd_tel[i] = std::round(control_a.gd(i)* 1000.0)/1000.0;
        status.pd_tel[i] = std::round(control_a.pd(i)* 1000.0)/1000.0;
        status.dm_piston[i] = std::round(control_u.dm_piston(i)* 1000.0)/1000.0;
        status.dl_offload[i] = std::round(next_offload(i)* 1000.0)/1000.0; // not the offload increment, but the total offload!
    }
    for (int i = 0; i < N_CP; i++) {
        status.closure_phase_K1[i] = std::round(bispectra_K1[i].closure_phase* 1000.0)/1000.0;
        status.closure_phase_K2[i] = std::round(bispectra_K2[i].closure_phase* 1000.0)/1000.0;
    }
    status.locked = control_u.fringe_found;
    // Count modulo 10000. This is mostly to look for skipped 
    // frames. 
    status.cnt = ft_cnt % 10000; 
    return status;
}

Settings get_settings() {
    Settings s;
    s.n_gd_boxcar = baselines.n_gd_boxcar;
    s.gd_threshold = gd_threshold;
    s.pd_threshold = pd_threshold;
    s.gd_search_reset = gd_search_reset;
    s.offload_time_ms = offload_time_ms;
    s.offload_gd_gain = pid_settings.offload_gd_gain;
    s.gd_gain = pid_settings.gd_gain * baselines.n_gd_boxcar;
    s.kp = pid_settings.kp;
    s.servo_mode = servo_mode;
    s.offload_mode = offload_mode;
    s.delay_line_type = delay_line_type;
    s.search_delta = control_u.search_delta;
    return s;
}

void test(uint beam, double value, int n) {
    // This is a test function that sets the DM piston to a value
    // and then waits for n seconds.
    if ((beam > N_TEL) || beam==0) {
        std::cout << "Beam number (arg 0) out of range (1 to " << N_TEL-1 << ")" << std::endl;
        return;
    }
    beam_mutex.lock();
    control_u.test_beam = beam - 1; // Convert to zero-indexed
    control_u.test_value = value;
    control_u.test_n = n;
    control_u.test_ix = 0;
    beam_mutex.unlock();   
}

// Zero the Group Delay offsets, once a fringe peak is found.
void zero_gd_offsets(void){
    baseline_mutex.lock();
    for (int bl=0; bl<N_BL; bl++)
        // Set the offsets to the group delay
        baselines.gd_phasor_offset(bl) = std::conj(baselines.gd_phasor(bl)); 
    baseline_mutex.unlock();
}

// Return the phasor offsets for all baselines to 3 decimal places
std::vector<double> get_gd_toml_offsets(void){
    std::vector<double> gd_offsets(6);
    baseline_mutex.lock();
    for (int bl=0;bl<N_BL; bl++)
        gd_offsets[bl] = std::round(std::arg(baselines.gd_phasor_offset(bl)) * gd_to_K1 * 1000)/1000.0;
    baseline_mutex.unlock();
    return gd_offsets;
}

// Set the parameters for the default fringe search. 
void set_search_params(double delta, uint turnaround){
    if (delta <= 0.0 || delta > 10.0){
        std::cout << "Search delta out of range (0.0 to 10.0 microns)" << std::endl;
        return;
    }
    if (turnaround < 1 || turnaround > 100){
        std::cout << "Search turnaround out of range (1 to 100 steps)" << std::endl;
        return;
    }
    beam_mutex.lock();
    control_u.search_delta = delta;
    control_u.steps_to_turnaround = turnaround;
    control_u.search_Nsteps = 0;
    beam_mutex.unlock();
    std::cout << "Search parameters updated: delta = " << delta << " microns, turnaround = " << turnaround << " steps" << std::endl;
}

bool foreground_in_place = false;

// state==1: apply offsets, state==0: reverse offsets
void set_foreground(int state) {
    static const Eigen::Vector4d fg_offset(-600.0, -200.0, 200.0, 600.0);
    if (state == 1 && !foreground_in_place) {
        if (offload_mode == OFFLOAD_OFF) offload_mode=OFFLOAD_MANUAL;
        add_to_delay_lines(fg_offset);
        foreground_in_place = true;
        
    } else if (state == 0 && foreground_in_place) {
    	if (offload_mode == OFFLOAD_MANUAL) offload_mode=OFFLOAD_OFF;
        add_to_delay_lines(-fg_offset);
        foreground_in_place = false;
    }
}

void delay_line_relative_move(double dl_move1, double dl_move2, double dl_move3, double dl_move4) {
    // Move the delay lines by a relative amount
    Eigen::Vector4d dl_move(dl_move1, dl_move2, dl_move3, dl_move4);
    add_to_delay_lines(dl_move);
}

void tweak_gd_offsets(double offset1, double offset2, double offset4) {
    // Add offsets to beams 1,2 and 4, then project onto baseline space
    Eigen::Vector4d tel_offsets = Eigen::Vector4d::Zero();
    tel_offsets(0) = offset1;
    tel_offsets(1) = offset2;
    tel_offsets(3) = offset4;
    Eigen::Matrix<double, N_BL, 1> bl_offsets = M_lacour * tel_offsets;
    baseline_mutex.lock();
    for (int bl = 0; bl < N_BL; bl++) {
        baselines.gd_phasor_offset(bl) *= std::exp(-1.0i * bl_offsets(bl));
    }
    baseline_mutex.unlock();
}

void set_gd_offsets(double offset1, double offset2, double offset4) {
    // Set offsets to beams 1, 2, and 4, then project onto baseline space
    Eigen::Vector4d tel_offsets = Eigen::Vector4d::Zero();
    tel_offsets(0) = offset1;
    tel_offsets(1) = offset2;
    tel_offsets(3) = offset4;
    Eigen::Matrix<double, N_BL, 1> bl_offsets = M_lacour * tel_offsets;
    baseline_mutex.lock();
    for (int bl = 0; bl < N_BL; bl++) {
        baselines.gd_phasor_offset(bl) = std::exp(-1.0i * bl_offsets(bl));
    }
    baseline_mutex.unlock();
}

// Set which beams are active (1=active, 0=inactive)
void beams_active(int b1, int b2, int b3, int b4) {
    beam_mutex.lock();
    if (b1==1) 
    	control_u.beams_active[0] = 1; 
    else 
    	control_u.beams_active[0] = 0;
    if (b2==1) 
    	control_u.beams_active[1] = 1; 
    else 
    	control_u.beams_active[1] = 0;
    if (b3==1) 
    	control_u.beams_active[2] = 1; 
    else 
    	control_u.beams_active[2] = 0;
    if (b4==1) 
    	control_u.beams_active[3] = 1; 
    else 
    	control_u.beams_active[3] = 0;
    beam_mutex.unlock();
    std::cout << "Active beams updated to: ";
    for (uint i = 0; i < N_TEL; i++) {
        std::cout << control_u.beams_active[i] << " ";
    }
    std::cout << std::endl;
}

void set_itime(double itime) {
    // Set the integration time in seconds
    if ((itime < 0) || itime>1000) {
        std::cout << "Target integration time out of range (0 to 1000)" << std::endl;
        return;
    }
    beam_mutex.lock();
    control_u.itime = 0;
    control_u.target_itime=itime;
    beam_mutex.unlock();   
    fmt::print("New integration started for a total time of {}\n", itime);
}

std::string expstatus(void){
    if (control_u.itime < control_u.target_itime) return "integrating";
    return "success";
}

std::string set_gd_boxcar(int n){
    if (n<1 || n>1000) return "ERROR: Boxcar out of range (1 to 1000)";
    baseline_mutex.lock();
    baselines.set_gd_boxcar(n);
    baseline_mutex.unlock();
    // Update the gd_gain to keep the same overall gain.
    pid_settings.mutex.lock();
    pid_settings.gd_gain = pid_settings.gd_gain * (double)baselines.n_gd_boxcar / (double)n;
    pid_settings.mutex.unlock();
    return "OK";
}

std::string default_gains(void){
    // Lets move maxmum of 70% of the way to the target in offload_time_ms. 
    double temp_gain = 0.0007 * offload_time_ms / (baselines.n_gd_boxcar * control_u.dit);
    if (temp_gain > 0.7) temp_gain = 0.7;
    baseline_mutex.lock();
    pid_settings.offload_gd_gain = temp_gain;     
    baseline_mutex.unlock();
    return "OK";
}

std::string set_dit(double dit){
    if (dit > 0.05) return "ERROR: DIT out of range (<0.05 seconds)";
    // Set the dit in the control structure
    control_u.dit = dit;
    return "OK";
}

std::string set_bad_pixels(std::vector<int> k1x, std::vector<int> k1y,
                          std::vector<int> k2x, std::vector<int> k2y) {
    // Set the bad pixels in the Fourier transform structures
    // Check that the input vectors are valid
    if (k1x.size() != k1y.size() || k2x.size() != k2y.size()) {
        return "ERROR: Bad pixel input vectors must have the same size";
    }
    // Check that the pixel coordinates are valid. They have to be smaller than subim_sz
    for (size_t i = 0; i < k1x.size(); i++) {
        if (k1x[i] < 0 || k1x[i] >= K1ft->subim_sz || k1y[i] < 0 || k1y[i] >= K1ft->subim_sz) {
                return "ERROR: Bad pixel coordinates out of range for K1";
        }
    }
    for (size_t i = 0; i < k2x.size(); i++) {
        if (k2x[i] < 0 || k2x[i] >= K2ft->subim_sz || k2y[i] < 0 || k2y[i] >= K2ft->subim_sz) {
                return "ERROR: Bad pixel coordinates out of range for K2";
        }
    }
    // Set the bad pixels
    K1ft->set_bad_pixels(k1x, k1y);
    K2ft->set_bad_pixels(k2x, k2y);
    return "OK";
}

COMMANDER_REGISTER(m)
{
    using namespace commander::literals;

    // You can register a function or any other callable object as
    // long as the signature is deductible from the type.
    m.def("linear_search", linear_search, "Execute a linear fringe search on a single beam (1,2,3 or 4)", 
        "beam"_arg, "start"_arg, "stop"_arg, "rate"_arg=1.0, "search_dt_ms"_arg=200, "search_snr_threshold"_arg=10.0);
    m.def("get_ps", get_ps, "Get the power spectrum in 2D", "filter"_arg="K1");
    m.def("get_search_offset", get_search_offset, "Get the search offset in microns");
    m.def("get_gd_toml_offsets", get_gd_toml_offsets, "Get the GD phasor offsets for all baselines in microns, to 3 decimal places");
    m.def("servo", set_servo_mode, "Set the servo mode", "mode"_arg="off");
    m.def("offload", set_offload_mode, "Set the offload (slow servo) mode", "mode"_arg="off");
    // Settings routines...
    m.def("offload_time", set_offload_time, "Set the offload time in ms", "time"_arg=1000);
    m.def("set_search_offset", set_search_offset, "Set the search offset in microns. \n This is added to the search position when starting a search.", 
        "offset"_arg=std::vector<double>(N_TEL, 0.0));
    m.def("dl", set_delay_line, "Set a delay line value in microns", 
        "beam"_arg, "value"_arg=0.0);
    m.def("dls", set_delay_lines_wrapper, "Set a delay line value in microns", 
        "dl1"_arg, "dl2"_arg, "dl3"_arg, "dl4"_arg);
        m.def("gain", set_gain, "Set the gain for the servo loop", "gain"_arg=0.0);
    m.def("ggain", set_ggain, "Set the gain for the GD servo loop", "gain"_arg=0.0);
    m.def("offload_gd_gain", set_offload_gd_gain, "Set the gain when operating GD only in steps", "gain"_arg=0.0);
    m.def("dl_type", set_delay_line_type, "Set the delay line type and initialize.", "type"_arg="piezo");
    m.def("set_gd_threshold", set_gd_threshold, "Set GD SNR threshold", "value"_arg=5.0);
    m.def("set_pd_threshold", set_pd_threshold, "Set PD SNR threshold", "value"_arg=4.5);
    m.def("set_gd_search_reset", set_gd_search_reset, "Set GD search reset threshold", "value"_arg=5.0);
    m.def("set_dit", set_dit, "Set the DIT in seconds", "dit"_arg=0.001);
    m.def("set_bad_pixels", set_bad_pixels, "Set the bad pixel map from 4 vectors", 
        "k1x"_arg=std::vector<int>(), "k1y"_arg=std::vector<int>(), "k2x"_arg=std::vector<int>(), "k2y"_arg=std::vector<int>());
    m.def("set_gd_boxcar", set_gd_boxcar, "Set the number of frames for the GD boxcar average", "n"_arg=32);
    m.def("tweak_gd_offsets", tweak_gd_offsets, "Add offsets to beams 1,2,4 and project to baseline space", 
        "offset1"_arg=0.0, "offset2"_arg=0.0, "offset4"_arg=0.0);
    m.def("set_gd_offsets", set_gd_offsets, "Set the GD offsets directly from a list of offsets for beams 1,2,4", 
        "offset1"_arg=0.0, "offset2"_arg=0.0, "offset4"_arg=0.0);
    m.def("beams_active", beams_active, "Set which beams are active", "b1"_arg=1,"b2"_arg=1,"b3"_arg=1,"b4"_arg=1);
    m.def("dlr", delay_line_relative_move, "Move the delay lines by a relative amount", 
        "dl_move1"_arg=0.0, "dl_move2"_arg=0.0, "dl_move3"_arg=0.0, "dl_move4"_arg=0.0);
    m.def("search", set_search_params, "Set the fringe tracker search parameter", 
        "delta"_arg=0.5, "turnaround"_arg=10);    
    // Special wag routines, and status.
    m.def("set_itime", set_itime, "Set the target integration time", "itime"_arg=100);
    m.def("status", get_status, "Get the status of the system");
    m.def("settings", get_settings, "Get current system settings");
    m.def("test", test, "Make a test pattern - fractional DM motion every n samples.", "beam"_arg, "value"_arg=0.0, "n"_arg=10);
    m.def("zero_gd_offsets", zero_gd_offsets, "Zero the group delay offsets i.e. track on this position");
    m.def("foreground", set_foreground, "Set (1) or unset (0) foreground delay line offsets", "state"_arg=1);
    m.def("expstatus", expstatus, "Get the exposure time status (success if complete)");
    m.def("default_gains", default_gains, "Set the gains to default values");
}

int main(int argc, char* argv[]) {
    IMAGE K1, K2;
    //Set the nice value.
    if (nice(-10)==-1) std::cout << "Re-niceing process likely didn't work. New nice value -1." << std::endl;
    // Read in the configuration file
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config file>.toml [options]" << std::endl;
        return 1;
    } else {
        config = toml::parse_file(argv[1]);
        std::cout << "Configuration file read: "<< config["name"] << std::endl;
    }

#ifndef SIMULATE
    // Initialise the DMs
    for (int i = 0; i < N_TEL; i++) {
        ImageStreamIO_openIm(&DMs[i], ("dm" + std::to_string(i+1) + "disp04").c_str());
        ImageStreamIO_openIm(&master_DMs[i], ("dm" + std::to_string(i+1)).c_str());
    }

    // Initialise the two forward Fourier transform objects
    ImageStreamIO_openIm(&K1, "hei_k1");
    ImageStreamIO_openIm(&K2, "hei_k2");
#else
    ImageStreamIO_openIm(&K1, "shei_k1");
    ImageStreamIO_openIm(&K2, "shei_k2");
    std::cout << "Simulation mode!" << std::endl;
   
#endif
    K1ft = new ForwardFt(&K1);
    K2ft = new ForwardFt(&K2);

    // Start the FFT threads
    K1ft->start();
    K2ft->start();

    // Start the main fringe-tracking thread. 
    std::thread fringe_thread(fringe_tracker);
    std::thread offloading_thread(dl_offload);
    
    // Initialize the commander server and run it
    commander::Server s(argc, argv);
    s.run();
    
    keep_offloading=false;
    offloading_thread.join();

    // Join the fringe-tracking thread
    servo_mode = SERVO_STOP;
    fringe_thread.join();

    // Join the FFTW threads. 
    //K1ft->stop();
    //K2ft->stop();
}
