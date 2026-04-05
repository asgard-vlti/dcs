#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <unistd.h>


/* ---------------------------------------------------------------------
   Program creating a thread set up with a high priority.

   Despite trying to play with the suid, I haven't been able to get access
   to the highest possible priorities (greater than 50 I think) unless I
   run this code as super user.

   This may be the missing piece to ensure more reliable data exchanges
   between the CRED1 and the PCI EDT board.
   --------------------------------------------------------------------- */

// max thread priority is 99
#define THREAD_PRIORITY 70 // Set desired thread priority

void* thread_function(void* arg) {
  while (1) {
    printf("Thread is running with high priority...\n");
    sleep(1);  // Simulate work
  }
  return NULL;
}

int main() {
  uid_t ruid; // real UID (user launching process at startup)
  uid_t euid; // effective UID (owner of executable at startup)
  uid_t suid; // saved UID
  int ret;

  getresuid(&ruid, &euid, &suid);
  ret = seteuid(ruid);  // normal user privileges
  printf("Real UID: %d, Effective UID: %d, Saved UID: %d\n", ruid, euid, suid);

  pthread_t thread;
  pthread_attr_t attr;
  struct sched_param param;
  int policy;
  
  // Initialize thread attributes
  pthread_attr_init(&attr);
  
  // Set the scheduling policy to SCHED_FIFO (or SCHED_RR)
  // this seems to be the part that requires super user privileges
  pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
  
  // Set the priority
  param.sched_priority = THREAD_PRIORITY;
  sched_setscheduler(0, SCHED_FIFO, &param);
  ret = seteuid(ruid);

  printf("Thread priority set to %d\n", param.sched_priority);
    
  if (pthread_attr_setschedparam(&attr, &param) != 0)
    perror("Failed to set sched parameters");
  printf("Scheduling parameters of thread were configured\n");
  
  // Create the thread with specified attributes
  if (pthread_create(&thread, &attr, thread_function, NULL) != 0) {
    perror("Failed to create thread");
    exit(EXIT_FAILURE);
  }
  printf("Thread successfully created\n");

  // Confirm the priority has been set by reading scheduling parameters
  pthread_getschedparam(thread, &policy, &param);
  
  printf("Thread priority set to %d\n", param.sched_priority);
  printf("Priority policy is %d\n", policy);
  
  // Clean up attribute object (can do this even if thread is still running)
  pthread_attr_destroy(&attr);
  
  // Join the thread (ie wait forever in this setup)
  pthread_join(thread, NULL);
  
  return 0;
}
