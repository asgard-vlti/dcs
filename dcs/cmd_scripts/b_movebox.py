#!/usr/bin/env python3

"""Move baldr camera ROI based on current baldr_tt peak position."""

import argparse
import sys

import dcs.ZMQutils


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move baldrN ROI to center using baldr_tt peak.")
    parser.add_argument("beam", type=int, help="Beam number (e.g. 1..4)")
    args = parser.parse_args()

    beam = args.beam
    if beam < 1:
        print("Beam must be >= 1", file=sys.stderr)
        return 2

    tt_port = 6670 + beam
    baldr_tt = dcs.ZMQutils.ZmqReq(f"tcp://192.168.100.2:{tt_port}")
    cam_server = dcs.ZMQutils.ZmqReq("tcp://192.168.100.2:6667")

    peak = baldr_tt.send_payload("peak", is_str=True, decode_ascii=False)
    if peak is None:
        print(f"No response from baldr_tt server on port {tt_port}", file=sys.stderr)
        return 1

    missing = [k for k in ("px_new", "py_new", "sz") if k not in peak]
    if missing:
        print(f"Invalid peak response, missing keys: {missing}. Response: {peak}", file=sys.stderr)
        return 1

    px_new = int(peak["px_new"])
    py_new = int(peak["py_new"])
    sz = int(peak["sz"])

    dx = px_new - (sz // 2)
    dy = py_new - (sz // 2)

    cmd = f'move_roi "baldr{beam}",{dx},{dy}'
    cam_server.s.send_string(cmd)
    response = cam_server.s.recv_string()

    print(f"peak: px_new={px_new}, py_new={py_new}, sz={sz}")
    print(f"sent: {cmd}")
    print(f"cam_server: {response.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
