#!/usr/bin/env python3
"""
A script that saves data from the shm /dev/shm/dmX.im.shm

Usage: python dm_telem.py --beam X --N 100

args:
--beam: the beam to track, e.g. dm1, dm2, etc. in [1,2,3,4]
--N: the number of frames to save in the cube (default: 100). If -1,
    save continuously until interrupted (e.g. with Ctrl+C) in cubes of
    100 frames each.
"""

import numpy as np
import astropy.io.fits as pf
from xaosim.shmlib import shm
import argparse
import sys
import datetime
import os


def main():
    parser = argparse.ArgumentParser(description="Saving DM telemetry datacube")
    parser.add_argument(
        "--beam",
        dest="beam",
        action="store",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="The beam to track (1, 2, 3, or 4)",
    )
    parser.add_argument(
        "--N",
        dest="nbframes",
        action="store",
        type=int,
        default=100,
        help="Number of images to save in cube (use -1 for continuous)",
    )
    args = parser.parse_args()

    target = f"/dev/shm/dm{args.beam}.im.shm"

    # Continuous mode: save in cubes of 100 frames each
    if args.nbframes == -1:
        cube_size = 100
        cube_count = 0
        try:
            while True:
                _save_cube(target, cube_size, args.beam, cube_count)
                cube_count += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit(0)
    else:
        _save_cube(target, args.nbframes, args.beam, 0)
        sys.exit()


def _save_cube(target, nbframes, beam, cube_count):
    now = datetime.datetime.utcnow()

    # --------
    stream = shm(target, nosem=False)
    ys, xs = stream.mtdata["y"], stream.mtdata["x"]
    dcube = np.zeros((nbframes, ys, xs))

    semid = 5
    stream.catch_up_with_sem(semid)

    if stream.mtdata["naxis"] == 3:
        for ii in range(nbframes):
            dcube[ii] = stream.get_latest_data_slice(semid)
    else:
        for ii in range(nbframes):
            dcube[ii] = stream.get_latest_data(semid)

    # --------
    sdir = f"/home/asg/data/{now.year}{now.month:02d}{now.day:02d}/dm_telem/"
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fname = sdir + f"dm{beam}_telem_{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    if cube_count > 0:
        fname += f"_cube{cube_count}"
    fname += ".fits"
    pf.writeto(fname, dcube, overwrite=True)
    print("wrote ", fname)


# -----------------------
if __name__ == "__main__":
    main()
