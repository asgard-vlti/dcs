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

// Configured at launch via config and updated by commander during runtime
CtrlSettings settings;

// Written to by servo_loop and read via commander interface
RTStatus rt_status;

// Mix of configurable values initialsed at launch and internal variables read/written
// by servo_loop.
ControlVariables ctrl;

// Image streams used by servo_loop
IMAGE DM_low;
IMAGE DM_high;
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

// Read the fits file containing the modes and store it in the provided Eigen matrix.
// The file should have N_MODES rows and N_ACTUATORS columns, but can have
// fewer than N_MODES rows, in which case the remaining modes will be set to zero.
bool read_modes(std::string filename, Eigen::Matrix<double, N_ACTUATORS, N_MODES> &modes)
{
  fitsfile *fptr; /* pointer to the FITS file, defined in fitsio.h */
  int status = 0; /* CFITSIO status value MUST be initialized to zero! */
  int nfound;
  long naxes[2] = {1, 1};
  double *data;

  if (fits_open_file(&fptr, filename.c_str(), READONLY, &status))
  {
    error("Error opening file: %s", filename.c_str());
    return false;
  }
  if (fits_read_keys_lng(fptr, "NAXIS", 1, 2, naxes, &nfound, &status))
  {
    error("Error reading NAXIS from file: %s", filename.c_str());
    return false;
  }
  if ((naxes[0] != N_ACTUATORS) || (naxes[1] > N_MODES))
  {
    error("Error: modes file has wrong dimensions. Expected %dx%d, got %ldx%ld", N_ACTUATORS, N_MODES, naxes[0], naxes[1]);
    return false;
  }
  data = new double[N_ACTUATORS * N_MODES];
  if (fits_read_img(fptr, TDOUBLE, 1, N_ACTUATORS * N_MODES, NULL, data, NULL, &status))
  {
    error("Error reading image data from file: %s", filename.c_str());
    delete[] data;
    return false;
  }
  // Copy the data into the Eigen matrix.
  for (long i = 0; i < N_MODES; i++)
  {
    for (long j = 0; j < N_ACTUATORS; j++)
    {
      if (naxes[1] > i)
      {
        modes(j, i) = data[i * N_MODES + j];
      }
      else
      {
        modes(j, i) = 0.0;
      }
    }
  }
  delete[] data;
  fits_close_file(fptr, &status);
  return true;
}

//----------commander functions from here---------------

Result load_reconstructor(std::string filename)
{
  // // This is a placeholder function for loading a reconstructor from a fits file.
  // // The actual implementation will depend on the format of the reconstructor file,
  // // which is not yet defined. For now, we will just print a message and return true.
  // warn("load_reconstructor unimplemented!");
  // info("Loading reconstructor from file: %s", filename.c_str());
  // fitsfile *fptr;
  // int status = 0, nkeys, ii;
  // long fpixel[2] = {1, 1};
  // info("opening file");
  // fits_open_file(&fptr, filename.c_str(), READONLY, &status);
  // info("reading pixels");
  // fits_read_pix(fptr, TDOUBLE, fpixel, N_PIXELS, NULL, ctrl.reconstructor.data(), NULL, &status);
  // return SUCCESS(true);
}

// Set the servo mode
Result set_servo_mode(std::string mode)
{
  int new_mode;
  if (mode == "off")
  {
    new_mode = SERVO_OPEN;
  }
  else if (mode == "lo")
  {
    new_mode = SERVO_CLOSED;
  }
  // TODO fix unimplemented:
  // else if (mode == "stop") {
  //   new_mode = SERVO_STOP;
  // }
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
  std::string msg = fmt::format("Servo mode updated to {}", new_mode);
  info(msg.c_str());
  return SUCCESS(msg);
}

// Set the tt gain.
Result set_log(double gain)
{
  settings.mutex.lock();
  settings.settings.log = gain;
  settings.mutex.unlock();
  return SUCCESS(settings.settings.log);
}

// Set the high order gain
Result set_hog(double gain)
{
  settings.mutex.lock();
  settings.settings.hog = gain;
  settings.mutex.unlock();
  return SUCCESS(settings.settings.hog);
}

// Set the high order leaky integrator term
Result set_hol(double leak)
{
  settings.mutex.lock();
  settings.settings.hol = leak;
  settings.mutex.unlock();
  return SUCCESS(settings.settings.hol);
}

// Set the tip/tilt leaky integrator term
Result set_lol(double leak)
{
  settings.mutex.lock();
  settings.settings.lol = leak;
  settings.mutex.unlock();
  return SUCCESS(settings.settings.lol);
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

Result poke_mode(int mode_ix, double amplitude)
{
  // ImAvgs im_avgs;
  // im_avgs.width = 0;
  // im_avgs.im_plus_sum_encoded = "";
  // im_avgs.im_minus_sum_encoded = "";
  // if (mode_ix < 0 || mode_ix >= N_MODES)
  // {
  //   std::string msg = fmt::format("Invalid mode index. Must be between 0 and %d", N_MODES - 1);
  //   info(msg.c_str());
  //   return FAILURE(msg);
  // }
  // // Encode the current im_plus_sum and im_minus_sum as base64 strings.
  // im_avgs.width = WIDTH;

  // // Set the control_u DM command to be the poke of the given mode and amplitude.
  // ctrl.modes.setZero();
  // ctrl.modes(mode_ix) = amplitude;
  // info("Poking mode %d with amplitude %f", mode_ix, amplitude);

  // // Wait 10ms for DM to settle, then set the im_plus_sum
  // // and im_minus_sum to zero.
  // usleep(10000);
  // im_mutex.lock();
  // for (size_t j = 0; j < WIDTH * WIDTH; j++)
  // {
  //   im_plus_sum[j] = 0;
  //   im_minus_sum[j] = 0;
  // }
  // im_mutex.unlock();

  // return SUCCESS(im_avgs);
}

COMMANDER_REGISTER(m)
{
  using namespace commander::literals;
  // You can register a function or any other callable object as
  // long as the signature is deductible from the type.
  m.def("servo", set_servo_mode, "Set the servo mode", "mode"_arg = "off");
  m.def("status", get_status, "Get the status of the system");
  m.def("settings", get_settings, "Get current system settings");
  m.def("log", set_log, "Set the low-order gain for the servo loop", "gain"_arg = 0.0);
  m.def("lol", set_lol, "Set the low-order leak term", "gain"_arg = 0.01);
  m.def("hog", set_hog, "Set the high-order gain for the servo loop", "gain"_arg = 0.0);
  m.def("hol", set_hol, "Set the high-order leak term", "gain"_arg = 0.01);
  m.def("focoff", set_focus_offset, "Set the focus offset", "offset"_arg = 0.0);
  m.def("pxy", set_pxy, "Set the origin pixels for tip/tilt", "px"_arg = 15, "py"_arg = 15);
  m.def("flux_threshold", set_flux_threshold, "Set flux threshold", "value"_arg = 100.0);
  m.def("poke", poke_mode, "Poke the DM with a given mode and amplitude", "mode_ix"_arg = 0, "amplitude"_arg = 0.1);
  m.def("recon", load_reconstructor, "Load a reconstructor from a fits file", "filename"_arg = "recon.fits");
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

  // Exit immediately if another instance of this server is running.
  char lockfile[256];
  sprintf(lockfile, "/tmp/asg.baldr_tt.%zu.lock", beam);
  if (!acquire_single_instance_lock(lockfile))
  {
    info("Another instance of this server is already running for beam %d. Exiting.", beam);
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
  settings.settings.log = config["log"].value_or(0.01);
  settings.settings.lol = config["lol"].value_or(0.01);
  settings.settings.hog = config["hog"].value_or(0.2);
  settings.settings.hol = config["hol"].value_or(0.01);
  settings.settings.flux_threshold = config["flux_threshold"].value_or(10000.0);
  settings.settings.servo_mode = SERVO_OPEN;
  // Read in the influence functions from the "modefile" fits file.
  std::string modefile = config["modefile"].value_or("modes.fits");
  // if (!read_modes(modefile, ctrl.projector))
  // {
  //   error("Error reading modes file. Exiting.");
  //   return 1;
  // }

  // Compute the rotation matrix R based on the rotation angle in the config file.
  double angle = config["dm_rotation"][beam - 1].value_or(0.0);

  errno_t err;
  bool anyerrors = false;

  const char *name = ("dm" + std::to_string(beam) + "disp01").c_str();
  err = ImageStreamIO_openIm(&DM_low, name);
  if (err != 0)
  {
    anyerrors = true;
    warn("failed to open shm: %s", name);
  }

  name = ("dm" + std::to_string(beam) + "disp02").c_str();
  err = ImageStreamIO_openIm(&DM_high, name);
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
  settings.settings.servo_mode = SERVO_STOP;
  servo_thread.join();

  unacquire_single_instance_lock();
  return 0;
}
