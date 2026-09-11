#!/usr/bin/env python3

"""Tweak BOTT/BOTP motors from the current baldr_tt peak position."""

import argparse
import sys

import zmq

import dcs.ZMQutils


def open_mds_connection() -> zmq.Socket:
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    socket.setsockopt(zmq.SNDTIMEO, 10000)
    socket.connect("tcp://192.168.100.2:5555")
    return socket


def send_and_get_response(socket: zmq.Socket, message: str) -> str:
    socket.send_string(message)
    response = socket.recv_string()
    return response.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adjust BOTT/BOTP from baldr_tt peak for faint-source alignment."
    )
    parser.add_argument("beam", type=int, help="Beam number (valid: 2..4)")
    args = parser.parse_args()

    beam = args.beam
    if beam not in (2, 3, 4):
        print("Beam must be 2, 3, or 4", file=sys.stderr)
        return 2

    tt_port = 6670 + beam
    baldr_tt = dcs.ZMQutils.ZmqReq(f"tcp://192.168.100.2:{tt_port}")
    peak = baldr_tt.send_payload("peak", is_str=True, decode_ascii=False)
    if peak is None:
        print(f"No response from baldr_tt server on port {tt_port}", file=sys.stderr)
        return 1

    missing = [key for key in ("px_new", "py_new", "sz") if key not in peak]
    if missing:
        print(f"Invalid peak response, missing keys: {missing}. Response: {peak}", file=sys.stderr)
        return 1

    px_new = int(peak["px_new"])
    py_new = int(peak["py_new"])
    sz = int(peak["sz"])

    dx = px_new - (sz // 2)
    dy = py_new - (sz // 2)

    x_offset = dx * 0.001
    y_offset = dy * 0.001

    mds = open_mds_connection()
    try:
        x_cmd = f"moverel BOTT{beam} {x_offset}"
        x_response = send_and_get_response(mds, x_cmd)

        y_cmd = f"moverel BOTP{beam} {y_offset}"
        y_response = send_and_get_response(mds, y_cmd)
    finally:
        mds.close(0)

    print(f"peak: px_new={px_new}, py_new={py_new}, sz={sz}")
    print(f"sent: {x_cmd}")
    print(f"mds: {x_response}")
    print(f"sent: {y_cmd}")
    print(f"mds: {y_response}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
