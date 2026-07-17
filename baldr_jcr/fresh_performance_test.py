import numpy as np
from supervisor import (  # type: ignore
    get_zmq_socket,
    DEFAULT_HOST,
    reset,
    create_polc_matrices,
    set_polc_gain,
    N_ACTX,
    N_ACTUATORS,
    DIST_LEN,
    update_com_dist_buffer,
    mode_telemetry,
    update_delay
)
import os
######################
# clean-slate reset all control params
socket = get_zmq_socket(beam=1, host=DEFAULT_HOST)
reset(socket, init=True)

######################
# set delay
socket = get_zmq_socket(beam=1, host=DEFAULT_HOST)
delay = float(os.environ.get("DELAY", 1.0))
update_delay(delay, socket)



#######################
# build polc controller with ALPHA, BETA, and NAVG specified
create_polc_matrices(
    socket,
    navg=int(os.environ.get("NAVG", 1)),
    alpha=float(os.environ.get("ALPHA", 1.0)),
    beta=float(os.environ.get("BETA", 1.0)),
    poke=float(os.environ.get("POKE", 1.0))
)
#########################
# turn on controller with specified GAIN and LEAK
set_polc_gain(
    float(os.environ.get("GAIN", 0.0)),
    float(os.environ.get("LEAK", 0.0)),
    socket=socket,
)
########################
# turn on disturbance
_, xx = np.meshgrid(
    np.linspace(0, 2 * 2 * np.pi, N_ACTX + 1)[:-1],
    np.linspace(0, 2 * 2 * np.pi, N_ACTX + 1)[:-1],
    indexing="ij",
)
xx_flat = xx.flatten()
disturbance = np.zeros([N_ACTUATORS, DIST_LEN])
for i, t in enumerate(np.linspace(0, 2 * np.pi, DIST_LEN + 1)[:-1]):
    disturbance[:, i] = 0.1 * np.sin(xx_flat + t)
update_com_dist_buffer(disturbance, socket=socket)
#####################
# run for (e.g.) 200 frames and print the average strehl for taguchi
os.system("rm /tmp/strehl")
mode_telemetry(200,0,socket)
with open("/tmp/strehl", "r") as f:
    lines = f.readlines()
# make sure there are at least 200 frames (for a valid experiment)
print(len(lines))
if len(lines) < 200:
    raise RuntimeError("something went wrong! couldn't get 200 samples!")
if len(lines) >= 300:
    raise RuntimeError("something went wrong! got too many samples!")
strehl = np.mean([float(l) for l in lines])
print(f"strehl:\n{strehl:0.7}")
