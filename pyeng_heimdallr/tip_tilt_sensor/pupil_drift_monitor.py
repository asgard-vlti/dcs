#!/usr/bin/env python3

from hmd_tts import HMD_TTS
from xaosim.shmlib import shm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

ddir = os.getenv("HOME") + "/Progs/repos/dcs/pyeng_heimdallr/tip_tilt_sensor/"
default_log = ddir + "log_pupil_monitor.log"

def log(message="", logfile=default_log, echo=True):
    """-----------------------------------------------------------------------
    Simple logging utility to keep track of HFO and HPOL modulation sequences
    -----------------------------------------------------------------------"""
    tstamp = datetime.utcnow().strftime("%D %H:%M:%S")
    myline = f"{tstamp}: {message}"
    with open(logfile, "a") as mylog:
        mylog.write(myline + "\n")
    if echo:
        print(myline, end="")

# ----------------------------------------------------------------------
def main():
    hmd = HMD_TTS()
    dstream = shm("/dev/shm/hei_k1.im.shm")
    im_offset = 1000

    img = dstream.get_data() - im_offset
    UVC = hmd.dense.kpi.UVC
    uu, vv = np.append(UVC[:,0], -UVC[:,0]), np.append(UVC[:,1], -UVC[:,1])

    cvis = hmd.get_raw_cvis(hmd.dense, img, full=True)
    v2 = np.abs(cvis)**2
    xy0 = hmd.sparse.kpi.VAC[:, :2]
    xy1 = hmd.infer_pupil_model(v2)

    # ----------------------
    f1, ax = plt.subplots()
    ax.scatter(uu, vv, c=v2, vmax=0.3)
    ax.grid(True)
    f1.set_size_inches(5,5, forward=True)
    f1.set_tight_layout(True)
    f1.savefig("powerspectrum.png")

    # ----------------------
    f2, ax = plt.subplots()
    ax.scatter(xy0[:,0], xy0[:,1], c='b', label="Theoretical")    
    ax.scatter(xy1[:,0], xy1[:,1], c='r', label="Measured")
    for ii in range(4):
        ax.text(xy0[ii,0]-0.2, xy0[ii,1]-0.2, f"B{ii+1}", fontsize=14)
    ax.legend(loc=1)
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.grid(True)
    f2.set_size_inches(5,5, forward=True)
    f2.set_tight_layout(True)
    f2.savefig("pupil.png")
    log(np.array2string(
        (xy1 - xy0[:,:2]).flatten(),
        suppress_small=True,
        floatmode='fixed',
        precision=3,
        separator=', '), echo=False)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

