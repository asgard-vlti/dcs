#!/usr/bin/env python3

import zmq
import time
import argparse

gain = 0.11 # HFO microns per HPOL ADU

# =========================================================================
def get_hpol_pos(socket, beamid=1):
    """Get the HPOL stepper motor position for the requested beam ID #"""    
    socket.send_string(f"read HPOL{beamid}")
    return int(socket.recv_string().strip())

# =========================================================================
def move_hpol(socket, pos, beamid=1):
    """Move the HPOL stepper motor to *pos*  the requested beam ID #"""
    socket.send_string(f"moveabs HPOL{beamid} {float(pos)}")
    socket.recv_string()

# =========================================================================
def get_dl_pos(socket, beamid=1):
    """Get delay line position (HFO) for the requested beam ID #"""
    socket.send_string(f"read HFO{beamid}")
    return float(socket.recv_string().strip()) * 1e3

# =========================================================================
def move_dl(socket, pos, beamid=1):
    """Move delay line (HFO) to *pos* for the requested beam ID #"""
    socket.send_string(f"moveabs HFO{beamid} {1e-3 * pos:.5f}")
    socket.recv_string()  # acknowledgement

# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined HPOL-HFO control")
    parser.add_argument("bid", type=int, help="Beam ID# (1-4)")
    parser.add_argument("pos", type=int, help="Desired HPOL position")
    args = parser.parse_args()
    if not (1 <= args.bid <= 4):
        parser.error("Invalid Beam ID #")
    return args


# ----------------------------------------------------------------------
def main():
    args = parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    socket.connect("tcp://192.168.100.2:5555")

    # figuring out the starting point
    # -------------------------------
    hpol0 = get_hpol_pos(socket, beamid=args.bid)
    hfo0 = get_dl_pos(socket, beamid=args.bid)
    print(f"Start HPOL{args.bid} = {hpol0:+04d}")
    print(f"Start HFO {args.bid} = {hfo0:.2f} um")

    hfo1 = hfo0 + (args.pos - hpol0) * gain
    move_hpol(socket, args.pos, beamid=args.bid)
    move_dl(socket, hfo1, beamid=args.bid)

    time.sleep(0.5)
    hpol = get_hpol_pos(socket, beamid=args.bid)
    hfo = get_dl_pos(socket, beamid=args.bid)
    print("---")
    print(f"Final HPOL{args.bid} = {hpol:+04d}")
    print(f"Start HFO {args.bid} = {hfo:.2f} um")

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
