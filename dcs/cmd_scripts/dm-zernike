#!/usr/bin/env python3

"""Set a single DM Zernike mode on disp channel 4 (shms[3]).

Example:
    dm-zernike 4 10 0.1
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.interpolate import griddata
from xaosim.pupil import _dist as dist
from xaosim.shmlib import shm
from xaosim.zernike import mkzer1


DMS = 12
APS = 10


def fill_mode(dmmap: np.ndarray, amask: np.ndarray) -> np.ndarray:
    """Extrapolate outside-aperture values using nearest-neighbor fill."""
    out = np.logical_not(amask)
    gx, gy = np.mgrid[0:DMS, 0:DMS]
    points = np.array([gx[amask], gy[amask]]).T
    values = np.array(dmmap[amask])
    grid_z0 = griddata(points, values, (gx[out], gy[out]), method="nearest")
    res = dmmap.copy()
    res[out] = grid_z0
    return res


def make_single_zernike_map(noll_index: int, amplitude: float) -> np.ndarray:
    """Match GUI Zernike map generation for one Noll mode."""
    dd = dist(DMS, DMS, between_pix=True)
    taper = np.exp(-(dd / 5.5) ** 20)
    amask = taper > 0.4

    zmode = mkzer1(noll_index, DMS, APS // 2, limit=False)
    if noll_index != 1:
        zmode -= zmode[amask].mean()
        zmode /= zmode[amask].std()
    zmode = fill_mode(zmode, amask)

    return amplitude * zmode


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dm-zernike",
        description="Set one DM Zernike map on shms[3] and trigger update.",
    )
    parser.add_argument("dm", type=int, help="DM number, e.g. 1..4")
    parser.add_argument("mode", type=int, help="Noll index, e.g. 10")
    parser.add_argument("amplitude", type=float, help="Mode amplitude, e.g. 0.1")
    args = parser.parse_args()

    if args.dm < 1:
        print("DM index must be >= 1", file=sys.stderr)
        return 2
    if args.mode < 1:
        print("Mode index must be >= 1", file=sys.stderr)
        return 2

    shmfs = np.sort(glob.glob(f"/dev/shm/dm{args.dm}disp*.im.shm"))
    shmf0 = f"/dev/shm/dm{args.dm}.im.shm"
    if len(shmfs) < 4:
        print(
            f"Expected at least 4 disp SHMs for dm{args.dm}, found {len(shmfs)}. DM server started?",
            file=sys.stderr,
        )
        return 1
    if not os.path.exists(shmf0):
        print(f"Missing shared memory root: {shmf0}", file=sys.stderr)
        return 1

    shms = []
    shm0 = None
    try:
        for path in shmfs:
            shms.append(shm(path, nosem=False))
        shm0 = shm(shmf0, nosem=False)

        zmap = make_single_zernike_map(args.mode, args.amplitude)
        shms[3].set_data(zmap)
        shm0.post_sems(1)

        print(
            f"Applied dm{args.dm}: zernike noll={args.mode}, amplitude={args.amplitude:.6g} -> shms[3]"
        )
        return 0
    finally:
        for s in shms:
            try:
                s.close(erase_file=False)
            except Exception:
                pass
        if shm0 is not None:
            try:
                shm0.close(erase_file=False)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
