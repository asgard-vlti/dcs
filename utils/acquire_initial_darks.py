#!/usr/bin/env python3

"""==================================================================== 

Having to manually request the acquisition of darks upon startup is
annoying. This command line is intended to make our life easier,
interacting with the MDS and the CRED1 to acquire initial darks for the
day.

==================================================================== """

import zmq
import time

context = zmq.Context()

cred1 = context.socket(zmq.REQ)
cred1.setsockopt(zmq.RCVTIMEO, 10000)
cred1.connect("tcp://192.168.100.2:6667")

# mds = context.socket(zmq.REQ)
# mds.setsockopt(zmq.RCVTIMEO, 10000)
# mds.connect("tcp://192.168.100.2:5555")

fpss = [1000, 500, 250] # a few camera frame rate settings
gains = [1, 5, 10, 20]  # a few camera gain settings

nap_time = 2  # timing for each acquisition

# memorize initial configuration of the camera
# --------------------------------------------

cred1.send_string("get_fps")
time.sleep(0.1)
start_fps = int(float(cred1.recv_string()))

time.sleep(0.1)

cred1.send_string("get_gain")
time.sleep(0.1)
start_gain = int(cred1.recv_string())

print(f"Initial settings: fps={start_fps} - gain={start_gain}")

# dark acquisition sequence
# -------------------------
for fps in fpss:
    print(f"FPS = {fps:4d} - GAIN = ", end="")
    cred1.send_string(f"set_fps {fps}")
    time.sleep(0.1)
    _ = cred1.recv_string()  # acknowledgement
    time.sleep(0.5)

    for gain in gains:
        print(f"{gain:3d} ", end="", flush=True)
        cred1.send_string(f"set_gain {gain}")
        time.sleep(0.1)
        _ = cred1.recv_string()  # acknowledgement
        time.sleep(0.5)

        cred1.send_string("make_dark")
        time.sleep(0.1)
        _ = cred1.recv_string()  # acknowledgement
        time.sleep(nap_time)
    print("")

cred1.send_string(f"set_fps {start_fps}")
time.sleep(0.1)
_ = cred1.recv_string()  # acknowledgement
time.sleep(nap_time)

cred1.send_string(f"set_gain {start_gain}")
time.sleep(0.1)
_ = cred1.recv_string()  # acknowledgement
time.sleep(nap_time)

print(f"Camera back to fps={start_fps} - gain={start_gain}")
