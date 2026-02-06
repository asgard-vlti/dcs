#!/usr/bin/env python3
from __future__ import annotations

import argparse
from baldr_python_rtc.baldr_rtc.server import main as server_main

beam2port = {
    1:"6662",
    2:"6663",
    3:"6664",
    4:"6665",
}
def _default_socket_for_beam(beam: int, host: str = "*", base_port: int = 3000) -> str:
    return f"tcp://{host}:{beam2port[beam]}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Baldr RTC Python server (per-beam instance).")
    ap.add_argument("--beam", type=int, required=True, choices=[1, 2, 3, 4], help="Beam/telescope index (1-4).")
    ap.add_argument("--phasemask", default="H3",choices=[f"H{ii}" for ii in [1,2,3,4,5]]+[f"J{ii}" for ii in [1,2,3,4,5]], help="ZWFS phasemask used. Index 1 is smallest, 5 is largest. H is H-band in Baldr, J is J-band in Baldr")
    ap.add_argument("--socket", default=None, help="Override commander REP endpoint, e.g. tcp://127.0.0.1:3001")
    ap.add_argument("--config", required=True, help="Path to config file (TOML).")
    ap.add_argument("--telem-dir", default=None, help="Telemetry output directory.")
    ap.add_argument("--telem-capacity", type=int, default=1000, help="Telemetry ring capacity (samples).")
    ap.add_argument("--flush-hz", type=float, default=1, help="Telemetry flush rate (Hz).")
    ap.add_argument("--chunk-seconds", type=float, default=0.1, help="Telemetry chunk duration (seconds).")
    ap.add_argument("--debug", action="store_true", help="Enable more verbose printing and/or plots for debugging")
    args = ap.parse_args()

    socket = args.socket or _default_socket_for_beam(args.beam)
    telem_dir = args.telem_dir or f"./telem/beam{args.beam}"

    return server_main(
        beam=args.beam,
        phasemask=args.phasemask,
        socket=socket,
        config_path=args.config,
        telem_dir=telem_dir,
        telem_capacity=args.telem_capacity,
        flush_hz=args.flush_hz,
        chunk_seconds=args.chunk_seconds,
        debug=args.debug,
    )


if __name__ == "__main__":
    raise SystemExit(main())


#copied real beam3 tioml from /usr/local/etc/baldr/rtc_config/ 
#/home/asg/Downloads/baldr3_rtc_test.toml
#baldr_python_rtc/scripts/baldr_server.py
#python baldr_python_rtc/scripts/baldr_server.py --beam 3 --phasemask H4 --config /home/asg/Downloads/baldr3_rtc_test.toml