#!/usr/bin/env python3

import numpy as np
import sys
from xaosim.shmlib import shm
import zmq
import time

n_frames = 200
#Threshold in terms of the multiple of the variance above which we 
#will flag a pixel as bad.
threshold = 5 

#--------------------------------------------------
#Connect to the Heimdallr server - we will give
#it the bad pixels. 
zmq_context = zmq.Context()
socket = zmq_context.socket(zmq.REQ)
socket.setsockopt(zmq.RCVTIMEO, 2000)
socket.connect("tcp://192.168.100.2:6660")

# Open the shared memory for the K1 and K2 subimages
K1 = shm("/dev/shm/hei_k1.im.shm", nosem=False)
K2 = shm("/dev/shm/hei_k2.im.shm", nosem=False)

#Loop every 10ms (about the slowest we'd run this at) and 
#accumulate the mean and square of the K1 and K2 subimages, 
#so that we can compute the variance and identify bad pixels.
sumK1 = np.zeros((K1.ny, K1.nx), dtype=np.float64)
sumK2 = np.zeros((K2.ny, K2.nx), dtype=np.float64)
sumK1sq = np.zeros((K1.ny, K1.nx), dtype=np.float64)
sumK2sq = np.zeros((K2.ny, K2.nx), dtype=np.float64)
for i in range(n_frames):
    imK1 = K1.get_latest_data() - 1000.0
    imK2 = K2.get_latest_data() - 1000.0
    sumK1 += imK1
    sumK2 += imK2
    sumK1sq += imK1**2
    sumK2sq += imK2**2
    time.sleep(0.01)
meanK1 = sumK1 / n_frames
meanK2 = sumK2 / n_frames
varK1 = sumK1sq / n_frames - meanK1**2
varK2 = sumK2sq / n_frames - meanK2**2

K1.close(erase_file=False)
K2.close(erase_file=False)

#For now, just print out the bad pixels. In the future, 
#we could send this to the Heimdallr server
var_threshold1 = np.median(varK1) * threshold
var_threshold2 = np.median(varK2) * threshold
badpixK1 = np.where(varK1 > var_threshold1)
badpixK2 = np.where(varK2 > var_threshold2)
print("Bad pixels in K1:", list(zip(badpixK1[0], badpixK1[1])))
print("Bad pixels in K2:", list(zip(badpixK2[0], badpixK2[1])))

socket.disconnect()
zmq_context.term()