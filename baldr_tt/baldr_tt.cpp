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
IMAGE DM;
IMAGE master_DM;
IMAGE subarray;
double *im_av, *im_plus, *im_minus;

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

Status get_status() {
    rt_status.mutex.lock();
    Status s = rt_status.s;
    rt_status.mutex.unlock();
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
    m.def("pxy", set_pxy, "Set the origin pixels for tip/tilt", "px"_arg=15, "py"_arg=15);
    m.def("flux_threshold", set_flux_threshold, "Set flux threshold", "value"_arg=100.0);
    m.def("zero_tt", zero_tt, "Zero tip/tilt based on current image position");
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
    settings.s.py = config["px"].value_or(15);
    width = config["width"].value_or(15);
    settings.s.gauss_hwidth = config["gauss_hwidth"].value_or(3.0);
    settings.s.ttg = config["ttg"].value_or(0.01);
    settings.s.ttl = config["ttl"].value_or(0.01);
    settings.s.hog = config["hog"].value_or(0.2); 
    settings.s.hol = config["hol"].value_or(0.01);
    settings.s.focus_amp = config["focus_amp"].value_or(0.02);

    // Compute the rotation matrix R based on the rotation angle in the config file. 
    double angle = config["rotation_angle"].value_or(0.0);
    double cos_angle = std::cos(angle * M_PI / 180.0);
    double sin_angle = std::sin(angle * M_PI / 180.0);
    control_u.R << cos_angle, -sin_angle, sin_angle, cos_angle;

#ifndef SIMULATE
    // Initialise the DM
    ImageStreamIO_openIm(&DM, ("dm" + std::to_string(beam) + "disp05").c_str()); //!!! Want 2 DMs...
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
