#define TOML_IMPLEMENTATION
#include "baldr_tt.h"
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
int beam=1, width=21;
PIDSettings settings;
RTStatus rt_status;
ControlU control_u;
ControlA control_a;
IMAGE DM_low;
IMAGE DM_high;
IMAGE master_DM;
IMAGE subarray;

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

// Read the fits file containing the modes and store it in the provided Eigen matrix. 
// The file should have N_MODES rows and N_ACTUATORS columns, but can have 
// fewer than N_MODES rows, in which case the remaining modes will be set to zero.
bool read_modes(std::string filename, Eigen::Matrix<double, N_ACTUATORS, N_MODES> &modes)
{
    fitsfile *fptr;   /* pointer to the FITS file, defined in fitsio.h */
    int status = 0;   /* CFITSIO status value MUST be initialized to zero! */
    int nfound, anynul;
    long naxes[2] = {1,1};
    double *data;

    if (fits_open_file(&fptr, filename.c_str(), READONLY, &status)) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }
    if (fits_read_keys_dbl(fptr, "NAXIS", 1, 2, naxes, &nfound, &anynul, &status)) {
        std::cerr << "Error reading NAXIS from file: " << filename << std::endl;
        return false;
    }
    if ((naxes[0] != N_ACTUATORS) || (naxes[1] > N_MODES)) {
        std::cerr << "Error: modes file has wrong dimensions. Expected " << N_ACTUATORS << "x" << N_MODES << ", got " << naxes[0] << "x" << naxes[1] << std::endl;
        return false;
    }
    data = new double[N_ACTUATORS*N_MODES];
    if (fits_read_img(fptr, TDOUBLE, 1, N_ACTUATORS*N_MODES, NULL, data, NULL, &status)) {
        std::cerr << "Error reading image data from file: " << filename << std::endl;
        delete[] data;
        return false;
    }
    // Copy the data into the Eigen matrix. 
    for (int i=0; i<N_MODES; i++) {
        for (int j=0; j<N_ACTUATORS; j++) {
            if (naxes[1] > i) {
                modes(j,i) = data[j*N_MODES + i];
            } else {
                modes(j,i) = 0.0;
            }
        }
    }
    delete[] data;
    fits_close_file(fptr, &status);
    return true;
}

//----------commander functions from here---------------

bool load_reconstructor(std::string filename){
    // This is a placeholder function for loading a reconstructor from a fits file. 
    // The actual implementation will depend on the format of the reconstructor file, 
    // which is not yet defined. For now, we will just print a message and return true.
    std::cout << "Loading reconstructor from file: " << filename << std::endl;
    return true;
}

// Set the servo mode
void set_servo_mode(std::string mode) {
    if (mode == "off") {
        servo_mode = SERVO_OFF;
    } else if (mode == "tt") {
        servo_mode = SERVO_TT;
    } else if (mode == "ho") {
        servo_mode = SERVO_HO;
    } else {
        std::cout << "Servo mode not recognised" << std::endl;
        return;
    }
    // Reset the control_u parameters !!! TODO
    std::cout << "Servo mode updated to " << servo_mode << std::endl;
    return;
}


// Set the tt gain.
void set_ttg(double gain) {
    settings.mutex.lock();
    settings.s.ttg = gain;
    settings.mutex.unlock();
}

// Set the high order gain
void set_hog(double gain) {
    settings.mutex.lock();
    settings.s.hog = gain;
    settings.mutex.unlock();
}

// Set the high order leaky integrator term
void set_hol(double leak) {
    settings.mutex.lock();
    settings.s.hol = leak;
    settings.mutex.unlock();
}

// Set the tip/tilt leaky integrator term
void set_ttl(double leak) {
    settings.mutex.lock();
    settings.s.ttl = leak;
    settings.mutex.unlock();
}

// Set the amplitude of the focus term
void set_focus_amp(double focus) {
    settings.mutex.lock();
    settings.s.focus_amp = focus;
    settings.mutex.unlock();
}

// Setter functions for thresholds. 
void set_flux_threshold(double val) { 
    settings.mutex.lock();
    settings.s.flux_threshold = val; 
    settings.mutex.unlock();
}

void set_pxy(int px_new, int py_new){
    // Check that the new px and py are more than width/2 from the edge, 
    // otherwise we might have problems with the Gaussian window.
    if (px_new < width/2 || px_new > sz - width/2 || py_new < width/2 || py_new > sz - width/2) {
        std::cout << "px and py must be between " << width/2 << " and " << sz - width/2 << std::endl;
        return;
    }
    // Set px and py!
    settings.mutex.lock();
    settings.s.px = px_new;
    settings.s.py = py_new;
    settings.mutex.unlock();
    // For debugging, print the new px and py.
    std::cout << "px and py updated to " << px_new << " " << py_new << std::endl;
}

void set_tto(double x, double y){
    settings.mutex.lock();
    settings.s.ttxo = x;
    settings.s.ttyo = y;
    settings.mutex.unlock();
}

void set_focus_offset(double offset){
    settings.mutex.lock();
    settings.s.focus_offset = offset;
    settings.mutex.unlock();
}

Status get_status() {
    rt_status.mutex.lock();
    Status s = rt_status.s;
    rt_status.mutex.unlock();
    s.tx = std::round(s.tx*1000)/1000.0;
    s.ty = std::round(s.ty*1000)/1000.0;
    s.flux = std::round(s.flux*10)/10.0;
    s.cnt = cnt % 10000; 
    return s;
}

Settings get_settings() {
    settings.mutex.lock();
    Settings s=settings.s;
    settings.mutex.unlock();
    return s;
}

std::string set_bad_pixels(std::vector<int> x, std::vector<int> y) {
    // Set the bad pixels are valid
    // Set the bad pixels
    return "OK";
}

void zero_tt(){
    // Based on the curent average subarr, find the maximum pixel and set
    // (px, py) to this. 
    int px_new = 0;
    int py_new = 0;
    double max = 0.0;
    for (int i=0;i<sz;i++){
        for (int j=0;j<sz;j++){
            if (im_av[i*sz+j] > max){
                max = im_av[i*sz+j];
                px_new = j;
                py_new = i;
            }
        }
    }
    // Set the new px and py. Error checking is done in set_pxy.
    set_pxy(px_new, py_new);
}

TTMet get_ttmet(unsigned int last_cnt){
    TTMet ttmet_vec;
    // Number of new ttmet values since last_cnt. Max is N_TTMET.
    int num_ttmet = (ttmet_save.cnt - last_cnt + N_TTMET) % N_TTMET; 
    // Fix the size of the returned vectors with ttmet_vec to num_ttmet.
    ttmet_vec.tx.resize(num_ttmet);
    ttmet_vec.ty.resize(num_ttmet);
    ttmet_vec.mx.resize(num_ttmet);
    ttmet_vec.my.resize(num_ttmet);
    // Fill the returned vectors with the most recent num_ttmet values.
    for (int i=0; i<num_ttmet; i++){
        int ix = (ttmet_save.cnt - num_ttmet + i) % N_TTMET;
        ttmet_vec.tx[i] = std::round(ttmet_save.tx[ix]*1000)/1000.0;
        ttmet_vec.ty[i] = std::round(ttmet_save.ty[ix]*1000)/1000.0;
        ttmet_vec.mx[i] = std::round(ttmet_save.mx[ix]*1000)/1000.0;
        ttmet_vec.my[i] = std::round(ttmet_save.my[ix]*1000)/1000.0;
    }
    ttmet_vec.cnt = ttmet_save.cnt;
    return ttmet_vec;
}

ImAvgs poke_mode(int mode_ix, double amplitude){
    if (mode_ix < 0 || mode_ix >= N_MODES) {
        std::cout << "Invalid mode index. Must be between 0 and " << N_MODES-1 << std::endl;
        return;
    }
    // Encode the current im_plus_sum and im_minus_sum as base64 strings.
    im_mutex.lock();
    std::string im_plus_sum_encoded = encode((char*)im_plus_sum, sizeof(float)*width*width);
    std::string im_minus_sum_encoded = encode((char*)im_minus_sum, sizeof(float)*width*width);
    im_mutex.unlock();
    
    ImAvgs im_avgs;
    im_avgs.width = width;
    im_avgs.im_plus_sum_encoded = im_plus_sum_encoded;
    im_avgs.im_minus_sum_encoded = im_minus_sum_encoded;

    // Set the control_u DM command to be the poke of the given mode and amplitude.
    control_u.DM.setZero();
    control_u.DM(mode_ix) = amplitude;
    std::cout << "Poking mode " << mode_ix << " with amplitude " << amplitude << std::endl;

    // Wait 10ms for DM to settle, then set the im_plus_sum 
    // and im_minus_sum to zero.
    usleep(10000);
    im_mutex.lock();
    for (int j=0;j<width*width;j++){
        im_plus_sum[j]=0;
        im_minus_sum[j]=0;
    }
    im_mutex.unlock();

    return im_avgs;
}

COMMANDER_REGISTER(m)
{
    using namespace commander::literals;
    // You can register a function or any other callable object as
    // long as the signature is deductible from the type.
    m.def("servo", set_servo_mode, "Set the servo mode", "mode"_arg="off");
    m.def("status", get_status, "Get the status of the system");
    m.def("settings", get_settings, "Get current system settings");
    m.def("ttg", set_ttg, "Set the tip/tilt gain for the servo loop", "gain"_arg=0.0);
    m.def("ttl", set_ttl, "Set the tip/tilt leak term", "gain"_arg=0.01);
    m.def("hog", set_hog, "Set the high-order gain for the servo loop", "gain"_arg=0.0);
    m.def("hol", set_hol, "Set the high-order leak term", "gain"_arg=0.01);
    m.def("focamp", set_focus_amp, "Set the amplitude of the focus term", "focus"_arg=0.0);
    m.def("focoff", set_focus_offset, "Set the focus offset", "offset"_arg=0.0);
    m.def("pxy", set_pxy, "Set the origin pixels for tip/tilt", "px"_arg=15, "py"_arg=15);
    m.def("tto", set_tto, "Set tip/tilt offsets", "tx"_arg=0, "ty"_arg=0);
    m.def("flux_threshold", set_flux_threshold, "Set flux threshold", "value"_arg=100.0);
    m.def("zero_tt", zero_tt, "Zero tip/tilt based on current image position");
    m.def("ttmet", get_ttmet, "Get the saved tip/tilt metrology", "last_cnt"_arg=0);
    m.def("poke", poke_mode, "Poke the DM with a given mode and amplitude", "mode_ix"_arg=0, "amplitude"_arg=0.1);
    m.def("recon", load_reconstructor, "Load a reconstructor from a fits file", "filename"_arg="recon.fits");
 }

int main(int argc, char* argv[]) {
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
    beam = config["beam"].value_or(2.05);
    settings.s.px = config["px"].value_or(15);
    settings.s.py = config["py"].value_or(15);
    width = config["width"].value_or(15);
    settings.s.gauss_hwidth = config["gauss_hwidth"].value_or(3.0);
    settings.s.ttg = config["ttg"].value_or(0.01);
    settings.s.ttl = config["ttl"].value_or(0.01);
    settings.s.hog = config["hog"].value_or(0.2); 
    settings.s.hol = config["hol"].value_or(0.01);
    settings.s.focus_amp = config["focus_amp"].value_or(0.02);
    settings.s.focus_offset = config["focus_offset"].value_or(0.0);
    settings.s.flux_threshold = config["flux_threshold"].value_or(100.0);
    // Read in the influence functions from the "modefile" fits file.
    std::string modefile = config["modefile"].value_or("modes.fits");
    if (!read_modes(modefile, control_a.influence_functions)) {
        std::cerr << "Error reading modes file. Exiting." << std::endl;
        return 1;
    }

    // Compute the rotation matrix R based on the rotation angle in the config file. 
    double angle = config["rotation_angle"].value_or(0.0);
    double cos_angle = std::cos(angle * M_PI / 180.0);
    double sin_angle = std::sin(angle * M_PI / 180.0);
    control_u.R << cos_angle, -sin_angle, sin_angle, cos_angle;
    std::cout << "R matrix: " << control_u.R(0,0) << control_u.R(0,1) << control_u.R(1,0) << control_u.R(1,1) << std::endl;

#ifndef SIMULATE
    // Initialise the DM
    ImageStreamIO_openIm(&DM_low, ("dm" + std::to_string(beam) + "disp01").c_str()); 
    ImageStreamIO_openIm(&DM_high, ("dm" + std::to_string(beam) + "disp02").c_str()); 
    ImageStreamIO_openIm(&master_DM, ("dm" + std::to_string(beam)).c_str());

    // Initialise the two forward Fourier transform objects
    ImageStreamIO_openIm(&subarray, ("baldr" + std::to_string(beam)).c_str());
#else
    ImageStreamIO_openIm(&subarray, "sbaldr1");
    std::cout << "Simulation mode!" << std::endl;
   
#endif
     // Start the main servo thread. 
    std::thread servo_thread(servo_loop);
    
    // Initialize the commander server and run it
    commander::Server s(argc, argv);
    s.run();

    // Join the fringe-tracking thread
    servo_mode = SERVO_STOP;
    servo_thread.join();
}
