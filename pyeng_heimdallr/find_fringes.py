#!/usr/bin/env python3

try:
    from wfs import Heimdallr
except:
    from pyeng_heimdallr.wfs import Heimdallr

import argparse
import math
import threading

# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Find fringes on all baselines")
    parser.add_argument("band", type=str, help="K1 or K2")
    parser.add_argument("srange", type=float,
                        help="+/- search range (> 0, in microns)")
    parser.add_argument("step", type=float,
                        help="search step (> 0, in microns)")
    parser.add_argument("-b", "--bid", type=int,
                        help="Beam ID (1,2,3,4) for single beam use case")
    args = parser.parse_args()
    if args.band.upper() not in ["K1", "K2"]:
        parser.error("Band is K1 or K2")
    if args.bid is not None:
        if not (1 <= args.bid <= 4):
            parser.error("Invalid Beam ID # (1,2,3 or 4)")
    args.srange = math.fabs(args.srange)
    args.step = math.fabs(args.step)
    return args

# ----------------------------------------------------------------------
def main():
    args = parse_args()
    print(args)

    wfs = Heimdallr()
    t = threading.Thread(target=wfs.loop)
    t.start()

    if args.bid is None:
        print("Scanning all HFOs")
        for bid in range(1, 5):
            wfs.fringe_search(
                bid,
                srange=args.srange,
                step=args.step,
                band=args.band.upper())
            
    else:
        print(f"Scanning HFO{args.bid} only")
        wfs.fringe_search(
            args.bid,
            srange=args.srange,
            step=args.step,
            band=args.band.upper())

    wfs.stop()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

