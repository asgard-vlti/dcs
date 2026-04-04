"""
Save an interaction matrix.
"""
import ZMQ_control_client as zmq_client
import json
import numpy as np
from astropy.io import fits
import sys
import time

if __name__ == "__main__":
    ims = []
    amp = 0.04
    _ = zmq_client.get_ims(f"poke 0,0")
    time.sleep(1)
    for mode in range(11):
        ims += [zmq_client.get_ims(f"poke {mode},{amp}")]
        time.sleep(1)
        ims += [zmq_client.get_ims(f"poke {mode},{-amp}")]
        time.sleep(1)
    ims += [zmq_client.get_ims(f"poke 0,0")]

    ims = np.array(ims)

    # Save the data to a fits file. 
    hdu = fits.PrimaryHDU()
    hdu.data = ims
    hdu.writeto("im.fits", overwrite=True)

