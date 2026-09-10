#define TOML_IMPLEMENTATION
#include "baldr.h"
#include <commander/commander.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <string>
#define SCHED_PRIORITY 70
// Commander struct definitions for json. This is in a separate file to keep the main code clean.
#include "commander_structs.h"

using namespace std::complex_literals;
extern "C"
{
#include <b64/cencode.h> // Base64 encoding, in C so Frantz can see how it works.
}
//----------Globals-------------------
// The input configuration
toml::table config;

///////  Servo parameters:

// Configured at launch via config
size_t beam = 1;
char *baldr_root;

// Configured at launch via config and updated by commander during runtime
CtrlSettings settings;

// Written to by servo_loop and read via commander interface
RTStatus rt_status;

// Mix of configurable values initialsed at launch and internal variables read/written
// by servo_loop.
ControlVariables ctrl;

// Image streams used by servo_loop
IMAGE DM_low;
IMAGE master_DM;
IMAGE subarray;

// Utility functions

// Based on https://sourceforge.net/p/libb64/git/ci/master/tree/examples/c-example1.c
// If bandwith is an issue, we could compress the data before encoding it.
std::string encode(const char *input, unsigned int size)
{
  /* set up a destination buffer large enough to hold the encoded data */
  // print the size of the input
  // std::cout << "Size of input: " << size << std::endl;
  // char* output = (char*)malloc(size*4/3 + 4); /* large enough */
  char *output = (char *)malloc(size * 2); /* large enough */
  /* keep track of our encoded position */
  char *c = output;
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

DEF_READ_CTRL_PARAM(meas_offset, measurement reference, N_PIXELS, 1, DOUBLE)
DEF_READ_CTRL_PARAM(flux_mask, mask for normalizing measurement, N_SUBARRAY_PIXELS, 1, DOUBLE)
DEF_READ_CTRL_PARAM(strehl_mask, mask for estimating strehl, N_SUBARRAY_PIXELS, 1, DOUBLE)
DEF_READ_CTRL_PARAM(meas_to_mode, reconstructor matrix, N_MODES, N_PIXELS, DOUBLE)
DEF_READ_CTRL_PARAM(filter_coeff_in, IIR input filter coefficients, N_MODES, FILTER_LEN, DOUBLE)
DEF_READ_CTRL_PARAM(filter_coeff_out, IIR output filter coefficients, N_MODES, FILTER_LEN, DOUBLE)
DEF_READ_CTRL_PARAM(mode_offset, mode offset vector, N_MODES, 1, DOUBLE)
DEF_READ_CTRL_PARAM(mode_max, maximum mode values(used in antiwinup), N_MODES, 1, DOUBLE)
DEF_READ_CTRL_PARAM(mode_min, minimum mode values(used in antiwinup), N_MODES, 1, DOUBLE)
DEF_READ_CTRL_PARAM(mode_to_com, modal projection matrix, N_ACTUATORS, N_MODES, DOUBLE)
DEF_READ_CTRL_PARAM(com_max, maximum command values(used in clipping), N_ACTUATORS, 1, DOUBLE)
DEF_READ_CTRL_PARAM(com_min, minimum command values(used in clipping), N_ACTUATORS, 1, DOUBLE)
#if DIST_LEN > 0
DEF_READ_CTRL_PARAM(com_dist_buffer, command disturbance buffer, N_ACTUATORS, DIST_LEN, DOUBLE)
#endif

Result reset_ctrl()
{
  ctrl.mutex.lock();
  ctrl.meas_raw.setZero();
  ctrl.meas_norm.setZero();
  ctrl.meas_cl.setZero();
  ctrl.mode_raw.setZero();
  ctrl.mode_filt.setZero();
  ctrl.com_raw.setZero();
  ctrl.com_clean.setZero();
  ctrl.com_write.setZero();
  ctrl.mode_raw_buffer.setZero();
  ctrl.mode_filt_buffer.setZero();
  ctrl.strehl_est = 0.5; // reset to a safe value.
  ctrl.flux_est = 1.0e8; // reset to an unlikely but safe value.
  ctrl.mutex.unlock();
  return SUCCESS();
}

// Set the servo mode
Result set_servo_mode(std::string mode)
{
  int new_mode;
  if (mode == "off")
  {
    new_mode = SERVO_OPEN;
  }
  else if (mode == "on")
  {
    new_mode = SERVO_CLOSED;
  }
  else
  {
    const char *msg = "Servo mode not recognised";
    info(msg);
    return FAILURE(msg);
  }
  settings.mutex.lock();
  settings.settings.servo_mode = new_mode;
  settings.mutex.unlock();
  // Reset the control_u parameters !!! TODO
  std::string msg = fmt::format("Servo mode updated to {}", mode);
  info(msg.c_str());
  return SUCCESS(msg);
}

// Setter functions for thresholds.
Result set_flux_threshold(double val)
{
  settings.mutex.lock();
  settings.settings.flux_threshold = val;
  settings.mutex.unlock();
  return SUCCESS(settings.settings.flux_threshold);
}

Result set_pxy(size_t px_new, size_t py_new)
{
  // Check that the new px and py are more than WIDTH/2 from the edge,
  // otherwise we might have problems with the Gaussian window.
  if (px_new < WIDTH / 2 || px_new > sz - WIDTH / 2 || py_new < WIDTH / 2 || py_new > sz - WIDTH / 2)
  {
    std::string msg = fmt::format("px and py must be between {} and {}", WIDTH / 2, sz - WIDTH / 2);
    info(msg.c_str());
    return FAILURE(msg);
  }
  // Set px and py!
  settings.mutex.lock();
  settings.settings.px = px_new;
  settings.settings.py = py_new;
  settings.mutex.unlock();
  // For debugging, print the new px and py.
  info("px and py updated to %d %d", px_new, py_new);
  return SUCCESS(std::vector({settings.settings.px, settings.settings.py}));
}

Result get_status()
{
  rt_status.mutex.lock();
  Status s = rt_status.status;
  rt_status.mutex.unlock();
  s.flux = std::round(s.flux * 10) / 10.0;
  s.cnt = cnt % 10000;
  return SUCCESS(s);
}

Result get_settings()
{
  settings.mutex.lock();
  Settings s = settings.settings;
  settings.mutex.unlock();
  return SUCCESS(s);
}

Result get_measurement_encoded()
{
  MeasBase64 meas;
  ctrl.mutex.lock();
  // Thanks to the mutex, we can guarantee that ctrl.cnt and ctrl.meas_norm
  // correspond to the same frame.
  meas.cnt = ctrl.cnt;
  meas.meas = encode((char *)ctrl.meas_norm.data(), sizeof(double) * N_PIXELS);
  ctrl.mutex.unlock();
  return SUCCESS(meas);
}

Result get_mode_encoded()
{
  ModeBase64 mode;
  ctrl.mutex.lock();
  mode.cnt = ctrl.cnt;
  mode.mode = encode((char *)ctrl.mode_filt.data(), sizeof(double) * N_MODES);
  ctrl.mutex.unlock();
  return SUCCESS(mode);
}

COMMANDER_REGISTER(m)
{
  using namespace commander::literals;
  // You can register a function or any other callable object as
  // long as the signature is deductible from the type.
  m.def("servo", set_servo_mode, "Set the servo mode", "mode"_arg = "off");
  m.def("reset", reset_ctrl, "Reset the ctrl internal parameters");
  m.def("status", get_status, "Get the status of the system");
  m.def("settings", get_settings, "Get current system settings");
  m.def("pxy", set_pxy, "Set the origin pixels", "px"_arg = 15, "py"_arg = 15);
  m.def("flux_threshold", set_flux_threshold, "Set flux threshold", "value"_arg = 100.0);
  m.def("meas", get_measurement_encoded, "Read meas_norm in Base64 encoding");
  m.def("mode", get_mode_encoded, "Read mode_filt in Base64 encoding");
  m.def("meas_offset", read_meas_offset, "Read meas_offset from file", "filename"_arg = "./baldr_jcr/meas_offset.fits");
  m.def("flux_mask", read_flux_mask, "Read flux_mask from file", "filename"_arg = "./baldr_jcr/flux_mask.fits");
  m.def("strehl_mask", read_strehl_mask, "Read strehl_mask from file", "filename"_arg = "./baldr_jcr/strehl_mask.fits");
  m.def("meas_to_mode", read_meas_to_mode, "Read meas_to_mode from file", "filename"_arg = "./baldr_jcr/meas_to_mode.fits");
  m.def("filter_coeff_in", read_filter_coeff_in, "Read filter_coeff_in from file", "filename"_arg = "./baldr_jcr/filter_coeff_in.fits");
  m.def("filter_coeff_out", read_filter_coeff_out, "Read filter_coeff_out from file", "filename"_arg = "./baldr_jcr/filter_coeff_out.fits");
  m.def("mode_offset", read_mode_offset, "Read mode_offset from file", "filename"_arg = "./baldr_jcr/mode_offset.fits");
  m.def("mode_max", read_mode_max, "Read mode_max from file", "filename"_arg = "./baldr_jcr/mode_max.fits");
  m.def("mode_min", read_mode_min, "Read mode_min from file", "filename"_arg = "./baldr_jcr/mode_min.fits");
  m.def("mode_to_com", read_mode_to_com, "Read mode_to_com from file", "filename"_arg = "./baldr_jcr/mode_to_com.fits");
  m.def("com_max", read_com_max, "Read com_max from file", "filename"_arg = "./baldr_jcr/com_max.fits");
  m.def("com_min", read_com_min, "Read com_min from file", "filename"_arg = "./baldr_jcr/com_min.fits");
#if DIST_LEN > 0
  m.def("com_dist_buffer", read_com_dist_buffer, "Read com_dist_buffer from file", "filename"_arg = "./baldr_jcr/com_dist_buffer.fits");
#endif
}

int main(int argc, char *argv[])
{
  // Read in the configuration file
  if (argc < 2)
  {
    info("Usage: %s <config file>.toml [options]", argv[0]);
    return 1;
  }
  else
  {
    config = toml::parse_file(argv[1]);
    info("Configuration file read: %s", config["name"].value_or("unknown"));
  }
  beam = config["beam"].value_or(1);
  baldr_root = std::getenv("BALDR_ROOT");
  if (baldr_root == NULL)
  {
    error("BALDR_ROOT environment variable not set. Exiting.");
    return 1;
  }

  // Exit immediately if another instance of this server is running.
  char lockfile[256];
  sprintf(lockfile, "/tmp/asg.baldr_tt.%zu.lock", beam);
  if (!acquire_single_instance_lock(lockfile))
  {
    error("Another instance of this server is already running for beam %d. Exiting.", beam);
    return 1;
  }

  settings.settings.px = config["px"].value_or(15);
  settings.settings.py = config["py"].value_or(15);
  // If /usr/local/etc/ttN.txt exists, override px and py with its values.
  {
    std::string tt_file = "/usr/local/etc/tt" + std::to_string(beam) + ".txt";
    std::ifstream ifs(tt_file);
    if (ifs.is_open())
    {
      int px_file, py_file;
      if (ifs >> px_file >> py_file)
      {
        info("Loaded px=%d py=%d from %s", px_file, py_file, tt_file.c_str());
        set_pxy(px_file, py_file);
      }
    }
  }
  settings.settings.flux_threshold = config["flux_threshold"].value_or(10000.0);
  settings.settings.servo_mode = SERVO_OPEN;

  // Now we initialise the servo control matrices/parameters.
  // We haven't spawned any other threads yet, so there is no need to
  // lock the mutex.

  // read all control matrices/vectors from fits files with same name.
  LOAD_FROM_FILE(meas_offset);
  LOAD_FROM_FILE(flux_mask)
  LOAD_FROM_FILE(strehl_mask)
  LOAD_FROM_FILE(meas_to_mode)
  LOAD_FROM_FILE(filter_coeff_in)
  LOAD_FROM_FILE(filter_coeff_out)
  LOAD_FROM_FILE(mode_offset)
  LOAD_FROM_FILE(mode_max)
  LOAD_FROM_FILE(mode_min)
  LOAD_FROM_FILE(mode_to_com)
  LOAD_FROM_FILE(com_max)
  LOAD_FROM_FILE(com_min)
#if DIST_LEN > 0
  LOAD_FROM_FILE(com_dist_buffer)
#endif

  errno_t err;
  bool anyerrors = false;
  const char *name = ("dm" + std::to_string(beam) + "disp01").c_str();
  err = ImageStreamIO_openIm(&DM_low, name);
  if (err != 0)
  {
    anyerrors = true;
    warn("failed to open shm: %s", name);
  }

  name = ("dm" + std::to_string(beam)).c_str();
  err = ImageStreamIO_openIm(&master_DM, name);
  if (err != 0)
  {
    anyerrors = true;
    warn("failed to open shm: %s", name);
  }

  name = ("baldr" + std::to_string(beam)).c_str();
  err = ImageStreamIO_openIm(&subarray, name);
  if (err != 0)
  {
    anyerrors = true;
    warn("failed to open shm: %s", name);
  }

  if (anyerrors)
  {
    error("Failed to open required shm files, exiting");
    return err;
  }

  // Start the main servo thread.
  std::thread servo_thread(servo_loop);

  // Prepare the thread parameters.
  struct sched_param param;
  int policy;
  param.sched_priority = SCHED_PRIORITY;

  // Set the K1ft, K2ft and fringe-tracking threads to real-time priority.
  pthread_setschedparam(servo_thread.native_handle(), SCHED_FIFO, &param);
  pthread_getschedparam(servo_thread.native_handle(), &policy, &param);
  info("Servo thread priority: %d  Priority policy: %d\n", param.sched_priority, policy);

  // Initialize the commander server and run it
  commander::Server s(argc, argv);
  s.run();

  // this code is typically uncreached, except when in "single-command" mode
  // or if the user changes the servo mode to servo stop via commander
  // join the servo thread
  servo_thread.join();

  unacquire_single_instance_lock();
  return 0;
}
