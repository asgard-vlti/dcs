/* =========================================================================
 * Asgard Multi DM control server - Frantz Martinache
 * 
 * The program mostly handles the creation, update and destruction of several
 * shared memory data structures (ImageStreamIO library by O. Guyon), that
 * are refered to as channels.
 * 
 * Once the start command is issued to the DM server shell, a thread monitors
 * the content of the different shared memory data structures, combines them
 * and then send an update command to the DM driver itself.
 * ========================================================================= */

#include <stdio.h>

#ifdef __GNUC__
#  if(__GNUC__ > 3 || __GNUC__ ==3)
#	define _GNUC3_
#  endif
#endif

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include <commander/commander.h>
#include <ImageStreamIO.h>

#include <BMCApi.h>  // the new API for the MultiDM!

/* =========================================================================
 *                            Global variables
 * ========================================================================= */
#define LINESIZE 256
#define CMDSIZE 200

int wxsz, wysz;          // window size
int ii;                  // dummy index value
IMAGE **shmarray = NULL; // shared memory img pointer (defined in ImageStreamIO.h)
int nch         = 5;     // number of DM channels (default = 5)
int dms         = 12;    // linear size of the DM in actuators
int nact        = 140;   // number of "real" actuators of the DM
int nvact       = 144;   // number of "virtual" acutators of the DM
int keepgoing   = 0;     // flag to control the DM update loop
int nch_prev    = 0;     // keep track of the # of channels before a change
char dashline[80] =
  "-----------------------------------------------------------------------------\n";

int ndm = 4; // the number of DMs to be connected
DM *hdms[4];  // the handles for the different deformable mirrors
BMCRC rv;    // result of every interaction with the driver (check status)
uint32_t *map_lut[4];  // the DM actuator mappings

int simmode = 0;  // flag to set to "1" to not attempt to connect to the driver
int timelog = 0;  // flag to set to "1" to log DM response timing
char drv_status[16] = "idle"; // to keep track of server status

// order to be reshuffled when reassembling the instrument
const char snumbers[4][BMC_SERIAL_NUMBER_LEN+1] = \
  {"17DW019#122", "17DW019#053", "17DW019#093", "17DW019#113"};

pthread_t tid_refresh;
pthread_t tid_loops[4];      // thread IDs for DM control loop
unsigned int targs[4] = {1, 2, 3, 4}; // thread integer arguments

/* =========================================================================
 *                       function prototypes
 * ========================================================================= */
int shm_setup();
void* dm_control_loop(void *_dmid);
void* dms_refresh(void *args);
double* map2D_2_cmd(double *map2D);
void MakeOpen(int dmid, DM* hdm);

/* =========================================================================
 *                           DM setup function
 * ========================================================================= */
void MakeOpen(int dmid, DM* hdm) {
  memset(hdm, 0, sizeof(DM));
  printf("Attempting to open device %s\n", snumbers[dmid-1]);
  rv = BMCOpen(hdm, snumbers[dmid-1]);
  
  if (rv) {
    printf("Error %d opening the driver type %u.\n", rv, (unsigned int)hdm->Driver_Type);
    printf("%s\n\n", BMCErrorString(rv));

    printf("Press any key to exit.\n");
    getc(stdin);
    exit(0);
  }
  printf("Opened Device %d with %d actuators.\n", hdm->DevId, hdm->ActCount);
  
  rv = BMCLoadMap(hdm, NULL, map_lut[dmid-1]);  // load the mapping into map_lut
}

/* =========================================================================
 *      Allocates shared memory data structures for the new settings
 * ========================================================================= */
int shm_setup() {
  int ii, kk;
  int shared = 1;
  int NBkw = 10;
  long naxis = 2;
  uint8_t atype = _DATATYPE_DOUBLE;
  uint32_t *imsize;
  char shmname[20];

  // shared memory representation of the DM is a 2D (12x12) map
  imsize = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
  imsize[0] = dms;
  imsize[1] = dms;

  if (shmarray != NULL) { // structure must be freed before reallocation!
    for (kk = 0; kk < ndm; kk++)
      for (ii = 0; ii < nch_prev; ii++) 
	ImageStreamIO_destroyIm(&shmarray[kk][ii]);
    free(shmarray);
    shmarray = NULL;
  }

  shmarray = (IMAGE**) malloc((ndm) * sizeof(IMAGE*));

  // allocates nch arrays (+ 1 for the combined channel)
  for (kk = 0; kk < ndm; kk++) {
    shmarray[kk] = (IMAGE*) malloc((nch+1) * sizeof(IMAGE));
  }

  for (kk = 0; kk < ndm; kk++) {
    // individual channels
    for (ii = 0; ii < nch; ii++) {
      sprintf(shmname, "dm%ddisp%02d", kk+1, ii); // root name of the shm
      ImageStreamIO_createIm_gpu(&shmarray[kk][ii], shmname, naxis, imsize, atype, -1,
				 shared, IMAGE_NB_SEMAPHORE, NBkw, MATH_DATA);
    }
    // the combined array
    sprintf(shmname, "dm%d", kk+1);              // root name of the shm
    ImageStreamIO_createIm_gpu(&shmarray[kk][nch], shmname, naxis, imsize, atype, -1,
			       shared, IMAGE_NB_SEMAPHORE, NBkw, MATH_DATA);
  }
  free(imsize);
  return 0;
}

/* =========================================================================
 *        Convert 2D DM map into the command to be sent to the DM
 *
 * 4 corner values of the 2D map must be dropped. Result array is 140 elts
 * ========================================================================= */
double* map2D_2_cmd(double *map2D) {
  int ii, jj;  // dummy indices
  double* cmd = (double*) malloc(nact * sizeof(double));
  // FILE* fd;

  jj = 0;  // index of value to add to the command
  for (ii = 0; ii < nvact; ii++) {  // iterate over "virtual" actuators
    if ((ii == 0) || (ii == 11) || (ii == 132) || (ii == 143))
      continue;
    else {
      cmd[jj] = map2D[ii];
      jj++;
    }
  }
  return cmd;
}

/* =========================================================================
 *                     DM surface control thread
 * ========================================================================= */
void* dm_control_loop(void *_dmid) {
  uint64_t cntrs[nch];
  int ii, kk;  // array indices
  double tmp_map[nvact];  // to store the combination of channels

  unsigned int dmid = *((unsigned int *) _dmid);

  for (ii = 0; ii < nch; ii++)
    cntrs[ii] = shmarray[dmid-1][nch].md->cnt0;  // init shm counters

  while (keepgoing > 0) {

    // all process must post semaphore 1
    ImageStreamIO_semwait(&shmarray[dmid-1][nch], 1);  // waiting for a DM update!

    for (ii = 0; ii < nch; ii++) {
      cntrs[ii] = shmarray[dmid-1][ii].md->cnt0; // update counter values
    }

    // -------- combine the channels -----------
    for (ii = 0; ii < nvact; ii++) {
      tmp_map[ii] = 0.0; // init temp sum array
      for (kk = 0; kk < nch; kk++) {
	tmp_map[ii] += shmarray[dmid-1][kk].array.D[ii];
      }
    }

    // ensure values are within acceptable (0, 1) range
    for (ii = 0; ii < nvact; ii++) {
      if (tmp_map[ii] < 0.0)
	tmp_map[ii] = 0.0;
      if (tmp_map[ii] > 1.0)
	tmp_map[ii] = 1.0;
    }

    // ------- update the shared memory ---------
    shmarray[dmid-1][nch].md->write = 1;   // signaling about to write
    for (ii = 0; ii < nvact; ii++) // update the combined channel
      shmarray[dmid-1][nch].array.D[ii] = tmp_map[ii];
    shmarray[dmid-1][nch].md->cnt1 = 0;
    shmarray[dmid-1][nch].md->cnt0++;
    // ImageStreamIO_sempost(&shmarray[dmid-1][nch], -1);
    shmarray[dmid-1][nch].md->write = 0;  // signaling done writing
  }
  return NULL;
}

/* =========================================================================
 *                       DMs refresh cycle thread
 * ========================================================================= */
void* dms_refresh(void *args) {
  double *cmd;
  struct timespec now; // clock readout
  FILE* fd;
  int kk;

  if (timelog)
    fd = fopen("speed_record.log", "w");  // Record DM interaction times

  while (keepgoing == 1) { // entering the DM update loop
    for (kk = 0; kk < ndm; kk++) {
      cmd = map2D_2_cmd(shmarray[kk][nch].array.D);
      rv = BMCSetArray(hdms[kk], cmd, map_lut[kk]);  // send cmd to DM
      if (rv) {
	printf("%s\n\n", BMCErrorString(rv));
      }
      free(cmd);
    }
    clock_gettime(CLOCK_REALTIME, &now);   // get time after issuing
    if (timelog)
      fprintf(fd, "%f\n", 1.0*now.tv_sec + 1e-9*now.tv_nsec);
  }
  if (timelog)
    fclose(fd); // closing the timing log file

  return NULL;
}

/* =========================================================================
 *            Functions registered with the commander server
 * ========================================================================= */

void start() {
  /* -------------------------------------------------------------------------
   *          Starts the monitoring of shared memory data structures
   * ------------------------------------------------------------------------- */
  int kk;

  if (keepgoing == 0) {
    keepgoing = 1; // raise the flag
    printf("DM control loop START\n");

    // trigger the shm monitoring threads
    for (kk = 0; kk < ndm; kk++) {
      pthread_create(&tid_loops[kk], NULL, dm_control_loop, &targs[kk]);
    }

    if (simmode != 1)
      pthread_create(&tid_refresh, NULL, dms_refresh, NULL);
    
  } else
    printf("DM control loop already running!\n");
  sprintf(drv_status, "%s", "running");

}

void stop() {
  /* -------------------------------------------------------------------------
   *            Stops the monitoring of shared memory data structures
   * ------------------------------------------------------------------------- */
  if (keepgoing == 1)
    keepgoing = 0;
  else
    printf("DM control loop already off\n");
  sprintf(drv_status, "%s", "idle");
}

std::string status() {
  /* -------------------------------------------------------------------------
   *                          Returns server status
   * ------------------------------------------------------------------------- */
  int nbshm = 24;
  int counter = 24;

  char fname[64];

  for (int ii = 0; ii < ndm; ii++) {
    sprintf(fname, "/dev/shm/dm%d.im.shm", ii+1);
    if (access(fname, F_OK) != 0) counter -=1;
  }

  for (int ii = 0; ii < ndm; ii++) {
    for (int jj = 0; jj < nch; jj++) {
      sprintf(fname, "/dev/shm/dm%ddisp%02d.im.shm", ii+1, jj);
      if (access(fname, F_OK) != 0) counter -=1;
    }
  }

  if (counter != nbshm)
      sprintf(drv_status, "%s", "SHM PROBLEM!");

  return drv_status;
}

int get_nch() {
  /* -------------------------------------------------------------------------
   *               Returns the number of virtual channels per DM
   * ------------------------------------------------------------------------- */
  return nch;
}

void set_nch(int ival) {
  /* -------------------------------------------------------------------------
   *               Updates the number of virtual channels per DM
   * ------------------------------------------------------------------------- */
  nch_prev = nch; // memory of the previous number of channels
  nch = ival;
  shm_setup();
  printf("Success: # channels = %d\n", ival);
}

void reset(int dmid, int channel) {
  /* -------------------------------------------------------------------------
   *                     Resets a DM channel (or all)
   * ------------------------------------------------------------------------- */

  double reset_map[nvact] = {0};  // reset command map 
  double *live_channel;

  if (dmid <= ndm) {
    if (dmid > 0) {
      if (channel < 0) {
	printf("Reset all virtual channels of DM %d!\n", dmid);
	for (int kk = 0; kk < nch; kk++) {
	  live_channel = shmarray[dmid-1][kk].array.D;  // live pointer
	  shmarray[dmid-1][kk].md->write = 1;  // signaling about to write
	  memcpy(live_channel,
		 (double *) reset_map,
		 sizeof(double) * nvact);
	  shmarray[dmid-1][kk].md->cnt0++;
	  ImageStreamIO_sempost(&shmarray[dmid-1][kk], -1);
	  shmarray[dmid-1][kk].md->write = 0;  // done writing
	}
      }
      else if (channel < nch) {
	printf("Reset virtual channel %d of DM %d\n", channel, dmid);
	live_channel = shmarray[dmid-1][channel].array.D;  // live pointer
	shmarray[dmid-1][channel].md->write = 1;  // signaling about to write
	memcpy(live_channel,
	       (double *) reset_map,
	       sizeof(double) * nvact);
	shmarray[dmid-1][channel].md->cnt0++;
	ImageStreamIO_sempost(&shmarray[dmid-1][channel], -1);
	shmarray[dmid-1][channel].md->write = 0;  // done writing
      }
      else {
	printf("Virtual channels 0-%d have been set-up!\n", nch);
      }
    }
  }
  else
    printf("Only %d DMs on Asgard\n", ndm);
}

void quit() {
  /* -----------------------------------------------------------------------
   *                       Clean exit of the program.
   * ----------------------------------------------------------------------- */
  int kk, ii;
  if (keepgoing == 1) stop();
  
  printf("DM driver server shutting down!\n");
    
  if (simmode != 1) {
    for (kk = 0; kk < ndm; kk++)
      rv = BMCClearArray(hdms[kk]);
    if (rv) {
      printf("%s\n\n", BMCErrorString(rv));
      printf("Error %d clearing voltages.\n", rv);
    }
    
    for (kk = 0; kk < ndm; kk++) {
      rv = BMCClose(hdms[kk]);
      if (rv) {
	printf("%s\n\n", BMCErrorString(rv));
	printf("Error %d closing the driver.\n", rv);
      }
      printf("%s\n\n", BMCErrorString(rv));
    }
    for (ii = 0; ii < ndm; ii++) {
      free(map_lut[ii]);
    }
  }
  if (shmarray != NULL) { // free the data structure
    for (kk = 0; kk < ndm; kk++) {
      for (ii = 0; ii < nch + 1; ii++) {
	ImageStreamIO_destroyIm(&shmarray[kk][ii]);
      }
    }
    free(shmarray);
    shmarray = NULL;
  }
  exit(0);
}


namespace co=commander;

COMMANDER_REGISTER(m) {
  using namespace co::literals;
  m.def("start", start, "Starts monitoring shared memory data structures.");
  m.def("stop", stop, "Stops monitoring shared memory data structures.");
  m.def("status", status, "Returns status of the DM server.");
  m.def("get_nch", get_nch, "Returns the number of virtual channels per DM.");
  m.def("set_nch", set_nch, "Updates the number of virtual channels per DM.");
  m.def("reset", reset, "Resets DM #arg_0 channel #arg_1 (or if arg_1=-1).");
  m.def("quit", quit, "Stops and closes the server.");  
}

/* =========================================================================
 *                                Main program
 * ========================================================================= */
int main(int argc, char **argv) {


  for (ii = 0; ii < ndm; ii++) {
    hdms[ii] = (DM *) malloc(sizeof(DM));
    map_lut[ii] = (uint32_t *) malloc(sizeof(uint32_t)*MAX_DM_SIZE);
  }

  if (simmode != 1)
    for (ii = 0; ii < ndm; ii++) {
      MakeOpen(ii+1, hdms[ii]);
      usleep(1000);
    }
  else {
    printf("Simulated DM scenario: the drivers are not connected\n");
    for (ii = 0; ii < ndm; ii++)
      printf("Simulated DM id = %d - serial number = %s.\n", ii+1, snumbers[ii]);
  }
  shm_setup();  // set up startup configuration
 

  // --------------------- set-up the prompt --------------------
  printf("%s", dashline);
  printf("    _    ____   ____    _    ____  ____        ____  __  __ \n");
  printf("   / \\  / ___| / ___|  / \\  |  _ \\|  _ \\      |  _ \\|  \\/  |\n");
  printf("  / _ \\ \\___ \\| |  _  / _ \\ | |_) | | | |_____| | | | |\\/| |\n");
  printf(" / ___ \\ ___) | |_| |/ ___ \\|  _ <| |_| |_____| |_| | |  | |\n");
  printf("/_/   \\_\\____/ \\____/_/   \\_\\_| \\_\\____/      |____/|_|  |_|\n");
  printf("%s", dashline);

  start();  // start the DM with the default number of channels
  co::Server s(argc, argv);    // start the commander server
  s.run();
  
  // --------------------------
  //  clean ending the program
  // --------------------------
  quit();
}
