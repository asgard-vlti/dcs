#!/usr/bin/env python3

try:
    from wfs import Heimdallr
except:
    from pyeng_heimdallr.wfs import Heimdallr

import argparse
import math
import threading

import dcs.ZMQutils

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

    mcs_client = dcs.ZMQutils.ZmqReq("tcp://192.168.100.2:7019")

    def send_and_recv_ack(self, msg):
        # recieve ack
        print(f"sending {msg}")
        resp = mcs_client.send_payload(msg, decode_ascii=False)
        if resp is None or resp.get("ok") == False:
            print(resp)
            print("Failed to send offsets to MCS")
        else:
            print("msg acked")

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

    msg = {
        "origin": "find_fringes",
        "data": [
            {"hdlr_complete": 1},
        ],
    }
    send_and_recv_ack(mcs_client, msg)

    wfs.stop()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

