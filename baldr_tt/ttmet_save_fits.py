"""
Connect to the socket and save a fixed number of tip/tilt metrology points
to a fits file (number as an optional argv input)

Data will come in as 4 json lists of variable length:
tx, ty, mx, my

The "cnt" variable is returned each time.
"""
import ZMQ_control_client as zmq_client
import json
import numpy as np
from astropy.io import fits
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_points = int(sys.argv[1])
    else:
        n_points = 4096

    print(f"Saving {n_points} tip/tilt metrology points to fits file.")
    tx_list = []
    ty_list = []
    mx_list = []
    my_list = []
    cnt = 0
    while (len(tx_list) < n_points):
        data = zmq_client.send(f"ttmet {cnt}")
        if (type(data) != dict):
        	raise UserWarning("Incorrect response: " + data);
        tx_list += data["tx"]
        ty_list += data["ty"]
        mx_list += data["mx"]
        my_list += data["my"]
        cnt = data["cnt"]
        print(f"Got point {len(tx_list):04d}/{n_points}, cnt={cnt}", end="\r")
        time.sleep(0.01)

    # Save the data to a fits file. 
    hdu = fits.PrimaryHDU()
    hdu.data = np.array([tx_list[:n_points], ty_list[:n_points], mx_list[:n_points], my_list[:n_points]])
    hdu.writeto("tt_metrology.fits", overwrite=True)

