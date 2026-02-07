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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_points = int(sys.argv[1])
    else:
        n_points = 1000

    print(f"Saving {n_points} tip/tilt metrology points to fits file.")
    tx_list = []
    ty_list = []
    mx_list = []
    my_list = []
    cnt = 0
    for i in range(n_points):
        resp = zmq_client.send("get_ttmet")
        data = json.loads(resp)
        tx_list.append(data["tx"])
        ty_list.append(data["ty"])
        mx_list.append(data["mx"])
        my_list.append(data["my"])
        cnt = data["cnt"]
        print(f"Got point {i+1}/{n_points}, cnt={cnt}", end="\r")

    # Save the data to a fits file. 
    hdu = fits.PrimaryHDU()
    hdu.header["N_POINTS"] = n_points
    hdu.data = np.array([tx_list, ty_list, mx_list, my_list])
    hdu.writeto("tt_metrology.fits", overwrite=True)

