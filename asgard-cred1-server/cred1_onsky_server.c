/* =========================================================================
 * Asgard camera control server - Frantz Martinache
 * 
 * The program mostly handles the creation, update and destruction of the
 * shared memory data structure (ImageStreamIO library by O. Guyon) used to
 * share frames acquired with the camera through the PCI EDT interface.
 *
 * The interactive prompt takes advantage of the commander library
 * 
 * Setting up the save mode so that only the 6 ROIs are saved and no longer
 * the full frame (still broadcast to shm) for engineering purposes
 * ========================================================================= */

#include <stdio.h>

#ifdef __GNUC__
#  if(__GNUC__ > 3 || __GNUC__ ==3)
#	define _GNUC3_
#  endif
#endif

//#define DEBUG_TIMING

#include <commander/commander.h> // commander header
#include <ImageStreamIO.h>       // libImageStreamIO header
#include <edtinc.h>              // EDT-PCI board API
#include <pthread.h>
#include <unistd.h>              // for access and usleep
#include <cjson/cJSON.h>         // for configuration file(s)
#include <fitsio.h>
#include <time.h>
#include <sys/stat.h>
#include <semaphore.h>

#include "cred1_cli_tools.h"     // command line interface (CLI) tools

/* =========================================================================
 *          Local data structure to keep track of camera settings
 * ========================================================================= */
typedef struct {		
  int FGchannel;          // frame grabber channel 
  uint32_t width, height, depth;
  uint32_t nbpix_frm;     // the number of pixels in a "frame"
  uint32_t nbpix_cub;     // the number of pixels in a "cube"
  uint32_t nbreads;       // the number of reads without detector reset
  uint32_t nbr_hlf;       // the number of reads saved in a FITS cube
  uint32_t bf_size;       // the number of frames in a cube
  int nfr_reset;          // # of frames affected by camera reset (in NDMR)
  int timeout;
  char cameratype[80];
  char status[16];        // camera overall status
  float maxfps;           // maximum frame rate
  float fps;              // current number of frames in Hz
  int gain = 1;           // camera gain

  // cropping parameters
  int cropmode = 0;       // 0: OFF, 1: ON
  int hei_max_row;        // highest Heimdallr useful pixel
  int bal_min_row;        // lowest Baldr useful pixel
  int nrows_cropped;      // number of rows cropped when reading

  int save_mode;          // flag to control the FITS data save mode
  int save_dark = 0;      // transient flag to trigger the FITS dark save
  int rt_dark_sub;        // flag to switch to real time dark subtract
  int ndmr_mode;          // non destructive readout mode flag
  char readmode[16];      // readout mode code -- "GCDS" or "NDMR"
  int valid_dark = 0;     // flag to track whether a valid dark is loaded
  char utdate[16];        // store the UT date to inform FITS header
  unsigned int offset = 1000; // unsigned int data needs offset
} CREDSTRUCT;

/* =========================================================================
 *   Local data structure to split the shm into other shm sub-arrays
 *        for the 4 baldr beams and the 2 heimdallr interferograms
 *        these are read/written to/from a json configuration file
 *         or eventually, a toml configuration file
 * 
 * each array stores: x0, y0, x1, y1 (in that order) coordinates in the
 * full image reference frame!
 * ========================================================================= */
typedef struct {
  char name[6] = "";     // name of the live image
  int x0, y0, xsz, ysz;  // corner coordinate and size of ROIs
  int nrs;               // number of readouts in a sequence
  long npx;              // number of pixels in ROI (= xsz * ysz)
  char _name[8] = "";    // name of the multiple reads data cube
  long nbs;              // number of frames saved per ROI data cube (x2)
  // int *tsig;          // time signature (ex: [2, -1, -1])
} subarray;

/* =========================================================================
 *                           function prototypes
 * ========================================================================= */

void* save_cube_to_fits(unsigned short *dcube, long naxes[3],
			char *fname, int bitpix);

// threads
void *fetch_imgs(void *arg);
// void *save_cube(void *_cube);
void* save_dark(void *arg);
void* save_roi_cubes(void *arg);

// configuration functions
void refresh_image_splitting_configuration();
int init_cam_configuration();
void optimize_cropping_parameters();
void shm_setup(int roi_too);
void free_shm(int roi_too);
char *read_json_file(const char *filename);
void update_dark();

// commander triggered functions
void fetch();
void stop();
void quit();
void update_fps(float fps);
void update_gain(int gain);
float query_fps();
float query_det_temp();
float query_water_temp();
int query_gain();
void set_save_mode(int _mode);
void trigger_save_dark();
void set_split_mode(int _mode);
void set_dark_sub_mode(int _mode);
void set_crop_mode(int _mode);
void set_ndmr_mode(unsigned int _mode);
void show_cam_conf();

/* =========================================================================
 *                            Global variables
 * ========================================================================= */
#define LINESIZE 256
#define CMDSIZE 50   // max ZMQ command size
#define OUTSIZE 200  // not sure a dedicated constant is warranted, but...


int splitmode = 0;   // flag to check if split mode is activated
int baud = 115200;   // serial connexion baud rate
EdtDev ed = NULL;    // handle for the serial connexion to camera CLI
PdvDev pdv_p = NULL; // handle for data camera link
int unit = 0;        // the PCI unit identifier

IMAGE *shm_img = NULL;         // shared memory img pointer
IMAGE *shm_img_dark = NULL;    // shared memory img dark pointer
IMAGE *shm_ROI_live = NULL;    // pointer to the different shared memory ROI

unsigned short *tosave = NULL; // holder for the cube to save to FITS
unsigned short *svdark = NULL; // holder for the dark to save to FITS

int **ROI_cubes = NULL;        // buffer for the ROI individual cubes
int **ROI_tosave = NULL;       // holder for the ROI cubes to save to FITS

int keepgoing = 0;             // flag to control the image fetching loop
long savecube_index = 0;       // to keep track of saved cubes
CREDSTRUCT *camconf;           // store relevant camera configuration options

pthread_t tid_fetch;           // thread ID for the image fetching to SHM
pthread_t tid_save;            // thread ID when saving data
pthread_t tid_save_dark;       // thread ID when saving a dark (triggered)
sem_t sync_save;               // semaphore to signal it's time to save!
char status_cstr[16] = "idle"; // to answer ZMQ status queries

char dashline[80] =
  "-----------------------------------------------------------------------------\n";

char savedir[100];  // the directory where the data will be saved

int nroi = 6; // number of regions of interest on the detector
int roi0 = 0; // the index of the first ROI to save (for skip_save_baldr_mode)

subarray *ROI = NULL;
int previous_timeouts = 0;

/* =========================================================================
 *                  JSON configuration file processing
 * ========================================================================= */

// ============================================================
char *read_json_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    rewind(file);

    char *data = (char *) malloc(length + 1);
    if (!data) {
        perror("Failed to allocate memory");
        fclose(file);
        return NULL;
    }

    fread(data, 1, length, file);
    data[length] = '\0';  // Null-terminate the string
    fclose(file);
    return data;
}

/* =========================================================================
 *      Split configuration is read (for now) from a JSON file.
 *   The procedure updates a shared readout configuration data structure.
 * ========================================================================= */
void refresh_image_splitting_configuration() {
  char split_conf_fname[LINESIZE];
  int ii;
  sprintf(split_conf_fname, "%s/.config/cred1_split.json", getenv("HOME"));
  if (access(split_conf_fname, F_OK) == 0) {
    printf("Configuration file %s found!\n", split_conf_fname);
    char *json_data = read_json_file(split_conf_fname);
    cJSON *json_root = cJSON_Parse(json_data);
    free(json_data);

    ii = 0;
    ROI = (subarray *) malloc(nroi * sizeof(subarray));
    // iterate over objects
    cJSON *item = NULL;
    cJSON_ArrayForEach(item, json_root) {
      const char *name = item->string;
      // printf("Name = %s\n", name);
      sprintf(ROI[ii].name, "%s", name);
      sprintf(ROI[ii]._name, "_%s", name);
      ROI[ii].x0  = cJSON_GetObjectItem(item, "x0")->valueint;
      ROI[ii].y0  = cJSON_GetObjectItem(item, "y0")->valueint;
      ROI[ii].xsz = cJSON_GetObjectItem(item, "xsz")->valueint;
      ROI[ii].ysz = cJSON_GetObjectItem(item, "ysz")->valueint;
      ROI[ii].nrs = 9; // 3; // cJSON_GetObjectItem(item, "nrs")->valueint; // assumes 3
      ROI[ii].npx = ROI[ii].xsz * ROI[ii].ysz;
      ROI[ii].nbs = 10000; // hardcoded here - should be fine
      ii++;
    }
    // update y0 coordinates of the Baldr ROI if cropmode is on
    if (camconf->cropmode == 1) {
      for (int ii = 0; ii < nroi; ii++) {
	if (strncmp(ROI[ii].name, "baldr", strlen("baldr")) == 0) {
	  ROI[ii].y0 -= camconf->nrows_cropped;
	}
      }
    }
  } else {
    printf("%s doesn't exist!\n", split_conf_fname);
    // split mode should not be possible here... do I want to bother?
    splitmode = 0;
  }
}

/* =========================================================================
 *       figure out the most aggressive possible cropping parameters
 * ========================================================================= */
void optimize_cropping_parameters() {
  camconf->bal_min_row = 255;  // reset values to initial
  camconf->hei_max_row = 0;    // 

  for (int ii = 0; ii < nroi; ii++) {
    if (strncmp(ROI[ii].name, "baldr", strlen("baldr")) == 0)
      if (ROI[ii].y0 <= camconf->bal_min_row){
		camconf->bal_min_row = ROI[ii].y0;
		printf("Found a lower baldr row %d\n", camconf->bal_min_row);
	}
    
    if (strncmp(ROI[ii].name, "hei_", strlen("hei_")) == 0)
      if (ROI[ii].y0 + ROI[ii].xsz >= camconf->hei_max_row){
		camconf->hei_max_row = ROI[ii].y0 + ROI[ii].xsz;
		printf("Found a higher hei row %d\n", camconf->hei_max_row);
	}
  }
  camconf->bal_min_row -= 2;
  camconf->hei_max_row += 16;

  printf("Cropping parameters are: %d and %d\n",
	 camconf->hei_max_row, camconf->bal_min_row);
}

/* =========================================================================
 *             initializes internal camera configuration data structure
 * ========================================================================= */
int init_cam_configuration() {
  char cmd_cli[CMDSIZE];  // holder for commands sent to the camera CLI
  // char out_cli[OUTSIZE];  // holder for CLI responses
  
  // ------
  camera_command(ed, "set cropping columns 1-10"); // on CRED1: groupings of 32

  // ------
  if (camconf->cropmode == 0) {
    camera_command(ed, "set cropping rows 1-256");
    camera_command(ed, "set cropping off");
    camconf->nrows_cropped = 0;
  }
  else {
    sprintf(cmd_cli, "set cropping rows 1-%d,%d-256",
	    camconf->hei_max_row, camconf->bal_min_row);
    camera_command(ed, cmd_cli);

    camera_command(ed, "set cropping on");
    camconf->nrows_cropped = camconf->bal_min_row - camconf->hei_max_row;

  }

  // ------
  camera_command(ed, "set mode globalresetcds"); // engineering mode
  camconf->ndmr_mode = 0;
  camconf->save_dark = 0;
  sprintf(camconf->readmode, "GCDS");

  // -----
  camconf->offset = 1000;
  camconf->bf_size = 200;
  camconf->nbreads = 200;
  camconf->nbr_hlf = 100;
  camconf->nfr_reset = 9; //UNUSED: to be made more adaptive !!!
  camconf->width = (uint32_t) pdv_get_width(pdv_p);
  camconf->height = (uint32_t) pdv_get_cam_height(pdv_p) - camconf->nrows_cropped;
  pdv_set_height(pdv_p, camconf->height);
  camconf->depth = pdv_get_depth(pdv_p);
  camconf->timeout = pdv_serial_get_timeout(ed);
  sprintf(camconf->cameratype, "%s", pdv_get_camera_type(pdv_p));
  
  camconf->nbpix_frm = camconf->width * camconf->height;
  camconf->nbpix_cub = camconf->nbpix_frm * camconf->nbr_hlf;

  //read_pdv_cli(ed, out_cli); // flush

  // camera_query_int(ed, "gain raw");
  camera_query_float(ed, "fps raw");
  camera_query_float(ed, "maxfps raw");
  camera_query_float(ed, "fps raw");

  camconf->gain = camera_query_int(ed, "gain raw");
  camconf->fps = camera_query_float(ed, "fps raw");
  camconf->maxfps = camera_query_float(ed, "maxfps raw");

  return 0;
}

/* =========================================================================
 *      Allocates shared memory data structures for the new settings
 * 
 * Depending on the value of roi_too, ROI shm will also created
 * ========================================================================= */
void shm_setup(int roi_too) {
  int shared = 1;
  int NBkw = 10;
  long naxis = 3;
  int nbpix = 0; // temporary variable to write concise lines
  uint32_t imsize[3] = {camconf->width, camconf->height, camconf->bf_size};
  uint32_t roisize[2];
  // uint32_t nd_roisize[3];  // for NDMR camera mode

  // =========================== primary SHM ===============================
  if (shm_img != NULL) {
    ImageStreamIO_destroyIm(shm_img);
    free(shm_img);
    shm_img = NULL;
  }
  shm_img = (IMAGE*) malloc(sizeof(IMAGE));
  ImageStreamIO_createIm_gpu(shm_img, "cred1", naxis, imsize, _DATATYPE_UINT16,
			     -1, shared, IMAGE_NB_SEMAPHORE, NBkw, MATH_DATA);

  // ========================= primary dark SHM =============================
  if (shm_img_dark != NULL) {
    ImageStreamIO_destroyIm(shm_img_dark);
    free(shm_img_dark);
    shm_img_dark = NULL;
  }
  shm_img_dark = (IMAGE*) malloc(sizeof(IMAGE));
  ImageStreamIO_createIm_gpu(shm_img_dark, "cred1_dark", naxis, imsize,
			     _DATATYPE_UINT16,
			     -1, shared, IMAGE_NB_SEMAPHORE, NBkw, MATH_DATA);

  // ================= memory chunk to save to disk ========================
  tosave = (unsigned short *) malloc(camconf->nbpix_cub *
				     sizeof(unsigned short));

  svdark = (unsigned short *) malloc(camconf->nbpix_cub * 2 *
				     sizeof(unsigned short));


  // =========================== create ROIs ===============================
  if (roi_too == 1) {
    // ================== the case of the "live" ROI =======================
    if (shm_ROI_live != NULL) {
      ImageStreamIO_destroyIm(shm_ROI_live);
      free(shm_ROI_live);
      shm_ROI_live = NULL;
    }

    naxis = 2;
    shm_ROI_live = (IMAGE*) malloc(sizeof(IMAGE) * nroi);
  
    for (int ii = 0; ii < nroi; ii++) {
      roisize[0] = ROI[ii].xsz;
      roisize[1] = ROI[ii].ysz;
      ImageStreamIO_createIm_gpu(&shm_ROI_live[ii], ROI[ii].name, naxis,
				 roisize, _DATATYPE_INT32, -1, shared,
				 IMAGE_NB_SEMAPHORE, NBkw, MATH_DATA);
    }
    // ================= memory chunk to save to disk ========================
    ROI_cubes = (int **) malloc(nroi * sizeof(int*));
    nbpix = ROI[0].npx * ROI[0].nbs;
    for (int ii = 0; ii < nroi; ii++) {
      ROI_cubes[ii] = (int *) malloc(nbpix * sizeof(int));
    }
    // ROI_tosave are half-sized temporary storage for the saving procedure
    ROI_tosave = (int **) malloc(nroi * sizeof(int*));
    nbpix /= 2;
    for (int ii = 0; ii < nroi; ii++) {
      ROI_tosave[ii] = (int *) malloc(nbpix * sizeof(int));
    }
  }
}

/* =========================================================================
 *   Free the memory allocated to the shared memory data structures
 * 
 * Depending on the value of roi_too, ROI shm are also deleted
 * ========================================================================= */
void free_shm(int roi_too) {
  int ii;
  // the main image
  if (shm_img != NULL) {
    ImageStreamIO_destroyIm(shm_img);
    free(shm_img);
    shm_img = NULL;
  }
  free(tosave);  // the holder for the data to save to disk
  free(svdark);

  // the dark for the main image
  if (shm_img_dark != NULL) {
    ImageStreamIO_destroyIm(shm_img_dark);
    free(shm_img_dark);
    shm_img_dark = NULL;
  }

  // the ROIs of the sub-images
  if (roi_too == 1) {
    if (shm_ROI_live != NULL) {
      for (ii= 0; ii < nroi; ii++) {
	ImageStreamIO_destroyIm(&shm_ROI_live[ii]);
	free(ROI_cubes[ii]);
	free(ROI_tosave[ii]);
      }
      // printf("Freeing that memory!");
      free(shm_ROI_live);
      shm_ROI_live = NULL;
      free(ROI_cubes);
      free(ROI_tosave);
    }
  }
}

/* =========================================================================
 *                  Save a memory block a fits data-cube
 * ========================================================================= */
void* save_cube_to_fits(void *dcube, long naxes[3],
			char *fname, int bitpix, int kw) {
  fitsfile *fptr;
  int status = 0;
  long naxis = 3;
  // long fpixel = 1; // first pixel to write
  long psize = naxes[0] * naxes[1] * naxes[2]; // cube size in pixels

  fits_create_file(&fptr, fname, &status);
  // fits_set_compression_type(fptr, RICE_1, &status);
  fits_create_img(fptr, bitpix, naxis, naxes, &status);

  if (bitpix == USHORT_IMG)
    fits_write_img(fptr, TUSHORT, 1, psize, (unsigned short *) dcube, &status);

  if (bitpix == LONG_IMG)
    fits_write_img(fptr, TINT, 1, psize, (int *) dcube, &status);

  // fill in the proper keywords
  if (kw > 0) {
    // fits_update_key(fptr, TLONG, "EXPOSURE", &tint, "Exposure time", &status);
    fits_update_key(fptr, TLONG, "CINDEX", &savecube_index, "Cube index", &status);
    fits_update_key(fptr, TSTRING, "UTDATE", &camconf->utdate, "UT Date this file was written", &status);
    fits_update_key(fptr, TFLOAT, "FPS", &camconf->fps, "Frame rate (Hz)", &status);
    fits_update_key(fptr, TINT, "GAIN", &camconf->gain, "Camera gain", &status);
    fits_update_key(fptr, TSTRING, "RO_MODE", &camconf->readmode, "Camera readout mode", &status);
    fits_update_key(fptr, TINT, "OFFSET", &camconf->offset, "DC offset to subtract when processing", &status);
  }
  fits_close_file(fptr, &status);
  return NULL;
}

/* =========================================================================
 *                  Save ROI data-cubes as fits (thread)
 * ========================================================================= */
void* save_roi_cubes(void *arg) {
  struct timespec tnow;  // time since epoch
  struct tm *uttime;     // calendar time
  struct stat st;
  long naxes[3];
  char fname[300];
  char tstamp[20];
  /*char CROP[5] = "FULL";  // or "CROP"
  if (camconf->cropmode == 1)
  sprintf(CROP, "%s", "CROP");*/

  // unsigned short *cube = (unsigned short*) _cube;

  while (camconf->save_mode == 1) {
    sem_wait(&sync_save); // blocking call, waiting for sem_post
    // get the time - to give the file a unique name
    clock_gettime(CLOCK_REALTIME, &tnow);  // elapsed time since epoch
    uttime = gmtime(&tnow.tv_sec);        // translate into calendar time

    sprintf(camconf->utdate, "%04d-%02d-%02d",
	    1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday);

    // figure out where to write to disk - and create the directory
    sprintf(savedir, "/data/%04d%02d%02d/",
	    1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday); 

    if (stat(savedir, &st) == -1)
      mkdir(savedir, 0700);

    // common time-stamp for all files saved this turn
    sprintf(tstamp, "T%02d:%02d:%02d.%03ld",
	    uttime->tm_hour, uttime->tm_min, uttime->tm_sec, tnow.tv_nsec/1000000);

    for (int ri = roi0; ri < nroi; ri++) {
      naxes[0] = ROI[ri].xsz;
      naxes[1] = ROI[ri].ysz;
      naxes[2] = ROI[ri].nbs / 2;
      //sprintf(fname, "%s/%s_%s.fits[compress Rice]", savedir, ROI[ri].name, tstamp);
      sprintf(fname, "%s/%s_%s.fits", savedir, ROI[ri].name, tstamp);
      save_cube_to_fits((void*) ROI_tosave[ri], naxes, fname, LONG_IMG, 1);
    }
  }
  return NULL;
}

/* =========================================================================
 *                                Save a dark
 * ========================================================================= */
void* save_dark(void *arg) {
  long naxes[3] = {camconf->width, camconf->height, camconf->bf_size};
  char fname[300];  // path + name of the fits file to write

  struct timespec tnow;  // time since epoch
  struct tm *uttime;     // calendar time
  struct stat st;

  char CROP[5] = "FULL";  // or "CROP"
  if (camconf->cropmode == 1)
    sprintf(CROP, "%s", "CROP");
 
  char NBFR[5];
  sprintf(NBFR, "%03d", camconf->bf_size);
  if (camconf->ndmr_mode == 1)
    sprintf(NBFR, "%03d", camconf->nbreads);
  
  // get the time - to give the file a unique name
  clock_gettime(CLOCK_REALTIME, &tnow);  // elapsed time since epoch
  uttime = gmtime(&tnow.tv_sec);        // translate into calendar time

  sprintf(camconf->utdate, "%04d-%02d-%02d",
	  1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday);

  sprintf(savedir, "/data/darks/%04d%02d%02d/",
	  1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday); 

  if (stat(savedir, &st) == -1)
    mkdir(savedir, 0700);

  // create file name
  sprintf(fname, "!%sdark_cred1_%s_%s_%s_fps_%04.0f_gain_%03d.fits",
	  savedir, CROP, camconf->readmode, NBFR,
	  camconf->fps, camconf->gain);

  save_cube_to_fits(svdark, naxes, fname, USHORT_IMG, 0);
  printf("%s was saved!\n", fname);
  camconf->valid_dark = 1;
  set_dark_sub_mode(1);
  return NULL;
}

/* =========================================================================
 *            Load the dark for the current mode into shared memory
 * ========================================================================= */
void update_dark() {
  fitsfile *fptr;
  int status = 0;
  char fname[300];  // path + name of the fits file to load
  struct stat st;
  unsigned short *new_dark = NULL; // holder for the dark

  struct timespec tnow;  // time since epoch
  struct tm *uttime;     // calendar time

  char CROP[5] = "FULL"; // or "CROP"
  if (camconf->cropmode == 1)
    sprintf(CROP, "%s", "CROP");
 
  char NBFR[5];
  sprintf(NBFR, "%03d", camconf->bf_size);
  if (camconf->ndmr_mode == 1)
    sprintf(NBFR, "%03d", camconf->nbreads);

  // get the time - to give the file a unique name
  clock_gettime(CLOCK_REALTIME, &tnow);  // elapsed time since epoch
  uttime = gmtime(&tnow.tv_sec);        // translate into calendar time
  
  new_dark = (unsigned short *) malloc(camconf->nbpix_cub * 2 *
				       sizeof(unsigned short));

  sprintf(camconf->utdate, "%04d-%02d-%02d",
	  1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday);

  sprintf(savedir, "/data/darks/%04d%02d%02d/",
	  1900 + uttime->tm_year, 1 + uttime->tm_mon, uttime->tm_mday); 

  sprintf(fname, "%sdark_cred1_%s_%s_%s_fps_%04.0f_gain_%03d.fits",
	  savedir, CROP, camconf->readmode, NBFR,
	  camconf->fps, camconf->gain);

  if (stat(fname, &st) == -1) {
    printf("%s does not exist\n", fname);
    camconf->valid_dark = 0;
    set_dark_sub_mode(0);
  }
  else {
    fits_open_file(&fptr, fname, READONLY, &status);
    fits_read_img(fptr, TUSHORT, 1, 2*camconf->nbpix_cub,
		  NULL, new_dark, NULL, &status);
    // direct fits_read_img to shared memory possible?
    memcpy(shm_img_dark->array.UI16,
	   (unsigned short *) new_dark,
	   sizeof(unsigned short) * 2*camconf->nbpix_cub);

    // read the dark and copy its content to shared memory
    fits_close_file(fptr, &status);
    printf("%s was loaded\n", fname);
    fflush(stdout);
    camconf->valid_dark = 1;
  }
  free(new_dark);
}

/* =========================================================================
 *                     Camera image fetching thread
 * ========================================================================= */
void* fetch_imgs(void *arg) {
#ifdef DEBUG_TIMING
  struct timespec tstart, tend;
  double dtim, dtother;
#endif
  uint8_t *image_p = NULL;
  unsigned short *liveimg_ptr; // pointer to latest image in circular buffer
  unsigned short *livedrk_ptr; // pointer to corresponding dark

  unsigned short *seq_img_ptr; // pointer used to build the "live" ROI
  int *liveroi_ptr;            // pointer to the "live" ROI

  int numbufs = 4; // was reduced from initial 256. To be further tested
  bool timeoutrecovery = false;
  int timeouts;
  long int liveindex = 0;
  long int liveroi_index = 0;

  int reset_cntr = 0, prev_reset_cntr = 0, skipped_frames = 0;
  int dark_reset_index = 0, matching_dark_index = 0;

  long nbpix_frm = camconf->nbpix_frm;
  long nbpix_cub = camconf->nbpix_cub;
  long nbpix_roi_tosave = ROI[0].npx * ROI[0].nbs / 2;

  int ii, jj, ri;  // ii,jj pixel indices, ri: ROI index
  int tsig[9] = {4,3,2,1,0,-1,-2,-3,-4};  // {2,-1,-1} 1to be dynamically allocated from the JSON. !!! If changing this, also change ROI[ii].nrs = 3; on line 230.
  int seq_indices[9] = {0};   // frame indices part of current time sequence
  // int prev_liveindex = 0;     // frame index of previous image

  int cam_xsz, roi_xsz, x0, y0;
  // unsigned int offset = 1000;  // kept here in case of bug with change (now in camconf)
  
  // ----- image fetching loop starts here -----
  cam_xsz = shm_img->md->size[0];  // camera frame size along the x-axis

  pdv_timeout_restart(pdv_p, true);
  pdv_flush_fifo(pdv_p);
  pdv_multibuf(pdv_p, numbufs);
  pdv_start_images(pdv_p, 0); // start a continuous acquisition

  
  while (keepgoing > 0) {
    timeoutrecovery = false;

	while (!timeoutrecovery) {
	  // ===================================================
      // before re-writing the circular buffer: save dark ?
      // ===================================================
      if (camconf->save_dark == 1) {
      	// save the entire circular buffer as a "dark"
	    memcpy(svdark, (unsigned short *) shm_img->array.UI16,
		  2 * nbpix_cub * sizeof(unsigned short));
	    pthread_create(&tid_save_dark, NULL, save_dark, NULL);
	    camconf->save_dark = 0;
	    if (camconf->ndmr_mode == 1) {
	      dark_reset_index = shm_img->md->size[2] - reset_cntr;
	      printf("Dark reset index = %d\n", dark_reset_index);
		}
		else {
	  	  dark_reset_index = 0;
	  	  printf("Filled the non-NDMR mode buffer.\n");
	  	  fflush(stdout);
		}
      }

#ifdef DEBUG_TIMING
      clock_gettime(CLOCK_REALTIME, &tstart);  // start time
#endif
      image_p = pdv_wait_images(pdv_p, 1); // read the image

      // ===================================================
      // read the reset marker in raw NDMR image
      // ===================================================
      if (camconf->ndmr_mode == 1) {
      	memcpy(&reset_cntr, image_p + 4, sizeof(unsigned int));
	if (reset_cntr - prev_reset_cntr > 1)
	  skipped_frames++;
	prev_reset_cntr = reset_cntr;
      	// printf("\rreset counter = %3d", reset_cntr);
      	// fflush(stdout);
      }

      // ===================================================
      // read the detector and update shared memory buffer
      // ===================================================
      liveimg_ptr = shm_img->array.UI16 + liveindex * nbpix_frm;
      
#ifdef DEBUG_TIMING
      clock_gettime(CLOCK_REALTIME, &tend);  // end time
      dtim = (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9;
      tstart = tend;
#endif
      pdv_start_images(pdv_p, numbufs);

      shm_img->md->write = 1;              // signaling about to write
      memcpy(liveimg_ptr,                  // copy image to shared memory
	     (unsigned short *) image_p,
	     sizeof(unsigned short) * nbpix_frm);

      // ===================================================
      //          are we subtracting a dark ?
      // ===================================================
      if (camconf->rt_dark_sub == 1) {
	if (camconf->ndmr_mode == 0)
	  livedrk_ptr = shm_img_dark->array.UI16 + liveindex * nbpix_frm;
	else {
	  // adapt index here. reset_cntr is the number of frames since the last reset,
	  // and dark_reset_index is the first index of the reset in the dark cube.
	  matching_dark_index = liveindex - reset_cntr + dark_reset_index;
	  matching_dark_index = matching_dark_index % shm_img->md->size[2];
	  livedrk_ptr = shm_img_dark->array.UI16 + matching_dark_index * nbpix_frm;
	}
	for (ii = 0; ii < nbpix_frm; ii++) // subtracting dark here
	  liveimg_ptr[ii] -= livedrk_ptr[ii] - camconf->offset;
      }

      // =============================
      //     SHM house-keeping for the main frame.
      // =============================      
      shm_img->md->write = 0;              // signaling done writing
      shm_img->md->cnt0++;                 // increment internal counter
      shm_img->md->cnt1 = shm_img->md->cnt0 % shm_img->md->size[2]; //Increment cyclical counter (for external SHM users)
      ImageStreamIO_sempost(shm_img, -1);  // post semaphores

      // =============================
      //   split into multiple ROIs
      // =============================      
      if (splitmode == 1) {
	for (ri = 0; ri < nroi; ri++) {
	  liveroi_ptr = shm_ROI_live[ri].array.SI32; // live ROI data pointer
	  roi_xsz = ROI[ri].xsz;
	  x0 = ROI[ri].x0;
	  y0 = ROI[ri].y0;
	  
	  shm_ROI_live[ri].md->write = 1;
	  // ------------ the "engineering" readout mode -------------
	  if (camconf->ndmr_mode == 0) {
	    for (jj = 0; jj < ROI[ri].ysz; jj++) {
	      for (ii = 0; ii < ROI[ri].xsz; ii++) {
		liveroi_ptr[jj*roi_xsz+ii] = liveimg_ptr[(jj+y0) * cam_xsz + ii+x0];
	      }
	    }
	  }
          // ------------- the "science" readout mode ----------------
          else {
	    // update frame indices from current sequence + previous index
            for (int kk = 0; kk < ROI[ri].nrs; kk++) {
              seq_indices[kk] = (shm_img->md->size[2] + liveindex - kk) % shm_img->md->size[2];
            }
	    // prev_liveindex = (seq_indices[0] - 1) % shm_img->md->size[2];
	    // ----- DETECTOR RESET -------
	    /*if (reset_cntr == 0) {
	      seq_img_ptr = shm_img->array.UI16 + prev_liveindex * nbpix_frm;  // live pointer
              for (jj = 0; jj < ROI[ri].ysz; jj++) {
                for (ii = 0; ii < ROI[ri].xsz; ii++) {
		  // just replicate previous image to avoid the blink
                  liveroi_ptr[jj*roi_xsz+ii] = (int)seq_img_ptr[(jj+y0) * cam_xsz + ii+x0];
                }
              }
            }
	    // ----- CLEAR of RESET -------
            else {*/
              for (jj = 0; jj < ROI[ri].ysz; jj++) {
                for (ii = 0; ii < ROI[ri].xsz; ii++) {
                  liveroi_ptr[jj*roi_xsz+ii] = camconf->offset;
                  for (int kk = 0; kk < ROI[ri].nrs; kk++) {
                    seq_img_ptr = shm_img->array.UI16 + seq_indices[kk] * nbpix_frm;  // live pointer
                    liveroi_ptr[jj*roi_xsz+ii] += tsig[kk] * (int)seq_img_ptr[(jj+y0) * cam_xsz + ii+x0];
                    // Mike's DEBUGing to see why liveindex was wrong !!! 
                    //if ((ri==4) && (jj==16) && (ii==16))
                    //	printf("liveindex %ld. kk %d seq_index %d pixel value %d\n", liveindex, kk, seq_indices[kk], (int)seq_img_ptr[(jj+y0) * cam_xsz + ii+x0]);
                  }
                }
              }
	      //}
	  }
          // ---------------- SHM house keeping ------------------
          shm_ROI_live[ri].md->write = 0;
          ImageStreamIO_sempost(&shm_ROI_live[ri], -1);
          shm_ROI_live[ri].md->cnt0++;
          shm_ROI_live[ri].md->cnt1++;

	  // -------------- writing to ROI cubes ----------------
	  liveroi_index = shm_img->md->cnt0 % ROI[ri].nbs;
	  memcpy(ROI_cubes[ri] + liveroi_index * ROI[ri].npx,
		 liveroi_ptr, ROI[ri].npx * sizeof(int));
        }
      }
#ifdef DEBUG_TIMING
      clock_gettime(CLOCK_REALTIME, &tend);  // end time
      dtother = (tend.tv_sec - tstart.tv_sec) + (tend.tv_nsec - tstart.tv_nsec) / 1e9;
      printf("Image %d: dtim = %.3f us, dtother = %.3f us\n",
	     liveindex, dtim * 1e6, dtother * 1e6);
#endif
      // =============================
      //    save the data to disk
      // =============================
      if ((camconf->save_mode == 1) && (splitmode==1)) {
	// ------------ the NEW intended save mode of the ROIs only -----------
	if (liveroi_index == ROI[0].nbs / 2) { // save the 1st half of the cubes
	  for (ri = roi0; ri < nroi; ri++) {
	    memcpy(ROI_tosave[ri], (int *) ROI_cubes[ri],
		   nbpix_roi_tosave * sizeof(int));
	  }
	  sem_post(&sync_save);  // signaling the thread it's time to save
	}
	if (liveroi_index == 0) { // save the 2nd half of the cubes
	  for (ri = roi0; ri < nroi; ri++) {
	    memcpy(ROI_tosave[ri], (int *) (ROI_cubes[ri] + nbpix_roi_tosave),
		   nbpix_roi_tosave * sizeof(int));
	  }
	  sem_post(&sync_save);  // signaling the thread it's time to save
	}
      }
      // ==============================
      // processing potential timeouts
      // ==============================
      timeouts = pdv_timeouts(pdv_p);
      if (timeouts > previous_timeouts){
        previous_timeouts = timeouts;
        timeoutrecovery = true;
        printf("Registering a camera timeout (current tally: %d)\n", timeouts);
      }
      
      if (keepgoing == 0)
	break;

      // Only update the liveindex at the very end	
      liveindex = shm_img->md->cnt0 % shm_img->md->size[2];

    }

    // =================
    // end of while loop
    // =================
  }
  return NULL;
}

/* =========================================================================
 *              Functions registered with the commander server
 * ========================================================================= */

/* -------------------------------------------------------------------------
 *          Trigger the thread fetching images to shared memory
 * ------------------------------------------------------------------------- */
void fetch() {
  if (keepgoing == 0) {
    keepgoing = 1; // raise the flag
    printf("Triggering the fetching data\n");
    pthread_create(&tid_fetch, NULL, fetch_imgs, NULL);
  }
  sprintf(status_cstr, "%s", "running");
}

/* -------------------------------------------------------------------------
 *            Processing Camera command-line interface exchanges.
 * ------------------------------------------------------------------------- */
std::string cli(std::string cmd) {
  char out_cli[OUTSIZE];  // holder for CLI responses  
  
  camera_command(ed, cmd.c_str());
  read_pdv_cli(ed, out_cli);
  return out_cli;
}

/* -------------------------------------------------------------------------
 *                          Returns server status
 * ------------------------------------------------------------------------- */
std::string status() {
  int nbshm = 8;
  int counter = 8;

  if (access("/dev/shm/cred1.im.shm", F_OK) != 0) counter -= 1;
  if (access("/dev/shm/cred1_dark.im.shm", F_OK) != 0) counter -= 1;
  if (access("/dev/shm/baldr1.im.shm", F_OK) != 0) counter -= 1;
  if (access("/dev/shm/baldr2.im.shm", F_OK) != 0) counter -= 1;
  if (access("/dev/shm/baldr3.im.shm", F_OK) != 0) counter -= 1;
  if (access("/dev/shm/baldr4.im.shm", F_OK) != 0) counter -= 1;
  if (access("/dev/shm/hei_k1.im.shm", F_OK) != 0) counter -= 1;
  if (access("/dev/shm/hei_k2.im.shm", F_OK) != 0) counter -= 1;

  if (counter != nbshm)
      sprintf(status_cstr, "%s", "SHM PROBLEM!");

  return status_cstr;
}

/* -------------------------------------------------------------------------
 *                      Interrupts the fetching process
 * ------------------------------------------------------------------------- */
void stop() {
  if (keepgoing == 1) {
    keepgoing = 0;
  }
  sprintf(status_cstr, "%s", "idle");
}

/* -------------------------------------------------------------------------
 *                            Quits the program
 * ------------------------------------------------------------------------- */
void quit() {
  if (keepgoing == 1) {
    keepgoing = 0;
    sleep(0.5);
  }
  free(ROI);
  free_shm(1); // erase everything, including ROI SHMs!
  exit(0);
}

/* -------------------------------------------------------------------------
 *           server level command to update the camera frame rate
 * ------------------------------------------------------------------------- */
void update_fps(float fps) {
  int wasrunning = 0;  // to keep track
  char cmd_cli[CMDSIZE];  // holder for commands sent to the camera CLI
  char out_cli[OUTSIZE];  // holder for CLI responses

  if (keepgoing == 1) {
    keepgoing = 0; // to interrupt the fetching process
    wasrunning = 1; //
  }
  sprintf(cmd_cli, "set fps %.2f", fps);
  camera_command(ed, cmd_cli);
  read_pdv_cli(ed, out_cli);

  camconf->fps = fps;
  update_dark();
  if (wasrunning == 1)
    fetch();
}

/* -------------------------------------------------------------------------
 * server level command to update the camera gain
 * currently, no value verification so don't be stupid
 * ------------------------------------------------------------------------- */
void update_gain(int gain) {
  int wasrunning = 0;  // to keep track
  char cmd_cli[CMDSIZE];  // holder for commands sent to the camera CLI
  char out_cli[OUTSIZE];  // holder for CLI responses

  if (keepgoing == 1) {
    keepgoing = 0; // to interrupt the fetching process
    wasrunning = 1; //
  }
  sprintf(cmd_cli, "set gain %d", gain);
  camera_command(ed, cmd_cli);
  read_pdv_cli(ed, out_cli);

  camconf->gain = gain;
  update_dark();
  if (wasrunning == 1)
    fetch();
}

/* -------------------------------------------------------------------------
 *           server level command to query the current camera frame rate
 * ------------------------------------------------------------------------- */
float query_fps() {
  float val = camera_query_float(ed, "fps raw");
  camconf->fps = val;
  return val;
}

/* -------------------------------------------------------------------------
 *           server level command to query the detector temperature
 * ------------------------------------------------------------------------- */
float query_det_temp() {
  float val = camera_query_float(ed, "temperature cryostat ptcontroller raw");
  return val;
}

/* -------------------------------------------------------------------------
 *           server level command to query the water temperature
 * ------------------------------------------------------------------------- */
float query_water_temp() {
  float val = camera_query_float(ed, "temperature water raw");
  return val;
}

/* -------------------------------------------------------------------------
 *           server level command to query the current camera gain
 * ------------------------------------------------------------------------- */
int query_gain() {
  int ival = camera_query_int(ed, "gain raw");
  camconf->gain = ival;
  return ival;
}

/* -------------------------------------------------------------------------
 * Start or interrupt the FITS saving of data cubes acquired by the camera
 * ------------------------------------------------------------------------- */
void set_save_mode(int _mode) {
  if (_mode <= 0) {
    camconf->save_mode = 0;
    printf("Savemode was turned OFF\n");
  }
  else {
    camconf->save_mode = 1;
    printf("Savemode was turned ON\n");
    pthread_create(&tid_save, NULL, save_roi_cubes, NULL);
  }
}

/* -------------------------------------------------------------------------
 * Setting to save only Heimdallr data streams
 * ------------------------------------------------------------------------- */
void skip_save_baldr_mode(int _mode) {
  if (_mode <= 0) {
    roi0 = 0;
  }
  else {
    roi0 = 4;
  }
}

/* -------------------------------------------------------------------------
 * Start or interrupt the FITS saving of data cubes acquired by the camera
 * 
 * ------------------------------------------------------------------------- */
void trigger_save_dark() {
  set_dark_sub_mode(0);
  // This next sleep has to be enough to fill a complete buffer (200 frames at 
  // time of writing).
  double min_sleep_time = 2.0*camconf->nbr_hlf/camconf->fps; 
  printf("Sleeping this thread for %6.3lf seconds...\n", min_sleep_time);
  sleep(min_sleep_time + 0.2);
  camconf->save_dark = 1;
  printf("Because Frantz and Mike aren't smart enough, sleeping again for %6.3lf seconds...\n", min_sleep_time);
  sleep(min_sleep_time + 0.2);
  fflush(stdout);
  set_dark_sub_mode(1);  
}

/* -------------------------------------------------------------------------
 *      Trigger the thread splitting the raw image into its different ROI
 * ------------------------------------------------------------------------- */
void set_split_mode(int _mode) {
  int wasrunning = 0;
  
  // if the acquisition is running, interrupt it and remember it was running
  if (keepgoing == 1) {
    keepgoing = 0;  // interrupt the acquisition
    wasrunning = 1; // keep that in mind
    sleep(1);
  }
  
  if (_mode <= 0) {
    splitmode = 0;
    printf("Splitmode was turned OFF\n");
  }
  else {
    splitmode = 1;
    printf("Splitmode was turned ON\n");

    free(ROI);
    refresh_image_splitting_configuration();
  }
  // back to prior business
  if (wasrunning == 1) {
    // sleep(0.2);
    fetch();
  }
}

/* -------------------------------------------------------------------------
 *     Switches the (speed optimized) cropped use mode of the CRED1 for
 * the Heimdallr/BALDR instrument setup. This will apply to the very next frame
 * saved in the cube in memory.
 * ------------------------------------------------------------------------- */
void set_dark_sub_mode(int _mode) {
  // update the RT dark subtract internal flag according to the command
  if (_mode <=0) {
    camconf->rt_dark_sub = 0;
    printf("No live dark subtraction\n");
  }
  else {
    update_dark();
    if (camconf->valid_dark == 1) {
      camconf->rt_dark_sub = 1;
      printf("Dark is live subtracted\n");
    }
    else {
      camconf->rt_dark_sub = 0;
      printf("No live dark subtraction\n");
    }
  }
}

/* -------------------------------------------------------------------------
 *     Switches the (speed optimized) cropped use mode of the CRED1 for
 * the Heimdallr/BALDR instrument setup.
 * ------------------------------------------------------------------------- */
void set_crop_mode(int _mode) {
  int wasrunning = 0;

  // if the acquisition is running, interrupt it and remember it was running
  if (keepgoing == 1) {
    keepgoing = 0;  // interrupt the acquisition
    wasrunning = 1; // keep that in mind
    sleep(1);
  }

  // update the cropmode internal flag according to the command
  if (_mode <=0) camconf->cropmode = 0;
  else camconf->cropmode = 1;

  // configure the detector readout
  optimize_cropping_parameters();   // figure out the rows to eliminate
  init_cam_configuration();         // configure the readout
  shm_setup(1);                     // reallocate main SHM memory

  update_dark();
  // back to prior business
  if (wasrunning == 1) {
    sleep(1);
    fetch();
  }
}

/* -------------------------------------------------------------------------
 *     Set up the CRED1 readout to non-destructive readout mode
 *
 * not fully functional yet and available for preliminary debugging.
 * implementation of time signatures for the different channels is missing
 * ------------------------------------------------------------------------- */
void set_ndmr_mode(unsigned int _mode) {
  char cmd_cli[CMDSIZE];  // holder for commands sent to the camera CLI
  char out_cli[OUTSIZE];  // holder for CLI responses

  // If the mode was 0 and was set to <=2, don't do anything.
  if ((camconf->ndmr_mode == 0) && (_mode <= 2))
    return;
  // If the mode was 1 and is set to the same number >2
  // as camconf->nbreads, don't do anything.
  if ((camconf->ndmr_mode == 1) && (_mode > 2) && (camconf->nbreads == _mode))
    return;

  int wasrunning = 0;
  
  // if the acquisition is running, interrupt it and remember it was running
  if (keepgoing == 1) {
    keepgoing = 0;  // interrupt the acquisition
    wasrunning = 1; // keep that in mind
    sleep(0.2);
  }

  if (_mode <= 2) {  // ------ engineering mode -----
    camconf->ndmr_mode = 0;
    sprintf(cmd_cli, "set mode globalresetcds");
    camera_command(ed, cmd_cli);
    read_pdv_cli(ed, out_cli);

    sprintf(cmd_cli, "set rawimages off");
    camera_command(ed, cmd_cli);
    read_pdv_cli(ed, out_cli);

    sprintf(camconf->readmode, "GCDS");
  }
  else {  // ------------------- science mode ------------------
    camconf->ndmr_mode = 1;
    sprintf(cmd_cli, "set rawimages on");
    camera_command(ed, cmd_cli);
    read_pdv_cli(ed, out_cli);
    sleep(0.1);

    sprintf(cmd_cli, "set mode globalresetbursts");
    camera_command(ed, cmd_cli);
    read_pdv_cli(ed, out_cli);
    sleep(0.1);
    
    camconf->nbreads = _mode;
    sprintf(cmd_cli, "set nbreadworeset %d", _mode); // _mode + 1 ??
    camera_command(ed, cmd_cli);
    read_pdv_cli(ed, out_cli);

    sprintf(camconf->readmode, "NDMR");
    camconf->nfr_reset = 9; // to be made more adaptive !!!
  }

  update_dark();
  // back to prior business
  if (wasrunning == 1) {
    sleep(0.2);
    fetch();
  }
}

/* =========================================================================
 *            Displays a summary of the camera configuration
 * ========================================================================= */
void show_cam_conf() {
  printf("\n%s", dashline);
  printf("Camera type       : %s\n", camconf->cameratype);
  printf("detector size     : %d x %d pixels\n",
	 pdv_get_width(pdv_p), pdv_get_height(pdv_p));

  printf("Frame size        : %d x %d pixels\n",
	 camconf->width, camconf->height);
  printf("Save cube size    : %d images\n",
	 camconf->nbr_hlf);
  printf("FPS               : %.2f Hz\n", camconf->fps);
  printf("Max-FPS           : %.2f Hz\n", camconf->maxfps);
  printf("detector gain     : %d\n", camconf->gain);
  printf("readout mode      : %s\n", camconf->readmode);
  if (strncmp(camconf->readmode, "NDMR", 4) == 0)
    printf("reads w/o reset   : %d\n", camconf->nbreads);
  if (camconf->rt_dark_sub > 0) printf("subtracting darks : YES\n");
  else printf("subtracting darks : NO\n");

  if (camconf->cropmode > 0) printf("crop-mode         : YES\n");
  else printf("crop-mode         : NO\n");
  printf("%s", dashline);
}

namespace co=commander;

COMMANDER_REGISTER(m)
{
  using namespace co::literals;
  m.def("fetch", fetch, "Trigger fetching data from the camera.");
  m.def("cli", cli, "Direct interface to the camera Command Line Interface.");
  m.def("status", status, "Get the current status of the camera.");
  m.def("stop", stop, "Stop fetching data from the. camera.");
  m.def("quit", quit, "Stops and closes the server.");
  m.def("set_fps", update_fps, "Updates the camera FPS and syncs SHM.");
  m.def("get_fps", query_fps, "Prints the current camera frame rate.");
  m.def("get_gain", query_gain, "Prints the current camera gain.");
  m.def("get_det_temp", query_det_temp, "Prints the detector temperature.");
  m.def("get_water_temp", query_water_temp, "Prints the water temperature.");
  m.def("set_gain", update_gain, "Updates the camera gain.");
  m.def("split_mode", set_split_mode, "Set/unset the multi ROI use mode.");
  m.def("save_mode", set_save_mode, "Set/unset the FITS cube save mode.");
  m.def("crop_mode", set_crop_mode, "Set/unset the cropped readout mode.");
  m.def("ndmr_mode", set_ndmr_mode, "Set/unset the multiple readout mode.");
  m.def("make_dark", trigger_save_dark, "Records dark for current config.");
  m.def("subtract_dark", set_dark_sub_mode, "Set/unset the dark subtraction.");
  m.def("cam_conf", show_cam_conf, "Summary of the current camera configuration");
  m.def("skip_save_baldr", skip_save_baldr_mode, "Skip saving BALDR data");
}

/* =========================================================================
 *                               Main program
 * ========================================================================= */
int main(int argc, char **argv) {
  char errstr[2*CMDSIZE];
  char edt_devname[CMDSIZE];

  sprintf(savedir, "/data/");  // cubes saved in ~/Data/ for now
  sem_init(&sync_save, 0, 0);  // saving semaphore now in place

  // ----- start with the serial CLI -----
  ed = pdv_open(EDT_INTERFACE, unit);
  if (ed == NULL) {
    edt_msg_perror(EDTAPP_MSG_WARNING, EDT_INTERFACE);
    return -1;
  }
  
  pdv_serial_read_enable(ed);
  pdv_serial_set_baud(ed, baud);
    
  // ----- and then, the data-link -----
  pdv_p = pdv_open(EDT_INTERFACE, unit);
  strcpy(edt_devname, EDT_INTERFACE);
  if (pdv_p == NULL) {
    sprintf(errstr, "pdv_open(%s, %d)", edt_devname, unit);
    edt_msg_perror(EDTAPP_MSG_WARNING, errstr);
    return (1);
  }
  pdv_flush_fifo(pdv_p);

  sleep(1);

  // --------------------- set-up the prompt --------------------
  
  printf("\n%s", dashline);
  printf(" _   _      _               _       _ _           ____        _     _      \n");
  printf("| | | | ___(_)_ __ ___   __| | __ _| | |_ __     | __ )  __ _| | __| |_ __ \n");
  printf("| |_| |/ _ \\ | '_ ` _ \\ / _` |/ _` | | | '__|____|  _ \\ / _` | |/ _` | '__|\n");
  printf("|  _  |  __/ | | | | | | (_| | (_| | | | | |_____| |_) | (_| | | (_| | |   \n");
  printf("|_| |_|\\___|_|_| |_|_|_|\\__,_|\\__,_|_|_|_|       |____/ \\__,_|_|\\__,_|_|   \n");
  printf("                 / ___|__ _ _ __ ___   ___ _ __ __ _                       \n");
  printf("                | |   / _` | '_ ` _ \\ / _ \\ '__/ _` |                      \n");
  printf("                | |__| (_| | | | | | |  __/ | | (_| |                      \n");
  printf("                 \\____\\__,_|_| |_| |_|\\___|_|  \\__,_|                      \n");
  printf("\n%s", dashline);

  // initial camera server setup
  camconf = (CREDSTRUCT*) malloc(sizeof(CREDSTRUCT));
  camconf->save_dark = 0;

  init_cam_configuration();
  refresh_image_splitting_configuration();
  set_crop_mode(1);  // now starting the CRED1 in crop mode
  update_fps(500.0); // engineering startup configuration
  update_gain(10);   // engineering startup configuration
  show_cam_conf();
  shm_setup(1);  // setup everything, including the ROI SHMs

  // --------------------------
  // start the commander server
  // --------------------------
  fetch();
  co::Server s(argc, argv);
  s.run();
  
  // ------------------------
  // clean-end of the program
  // ------------------------
  if (keepgoing == 1) {
    keepgoing = 0;
    sleep(0.5);
  }
  sem_destroy(&sync_save);  // saving semaphore erased
  printf("%s\n", status_cstr);
  free(ROI);
  free_shm(1); // erase everything, including ROI SHMs!
  exit(0);
}
