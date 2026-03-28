#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zmq


def main() -> int:
    ap = argparse.ArgumentParser(description="Commander client (ZMQ REQ).")
    ap.add_argument("--socket", required=True, help="Commander endpoint, e.g. tcp://127.0.0.1:3001")
    ap.add_argument("message", nargs="+", help='Command string, e.g. "status" or \'readBDRConfig "./cfg.toml"\'')

    args = ap.parse_args()
    msg = " ".join(args.message).strip()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(args.socket)
    sock.send_string(msg)
    reply = sock.recv_string()

    try:
        obj = json.loads(reply)
        print(json.dumps(obj, indent=2))
    except Exception:
        print(reply)

    sock.close(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
