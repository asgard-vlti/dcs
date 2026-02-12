from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import shlex

# to start all the Baldr RTCs
BEAMS = [1, 2, 3, 4]


def launch_in_xterm(title: str, argv: list[str]):
    # Build the command string safely
    cmd_str = " ".join(shlex.quote(x) for x in argv)

    # -hold keeps terminal open after exit (important for debugging)
    subprocess.Popen([
        "xterm",
        "-T", title,
        "-hold",
        "-e", cmd_str,
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phasemask", default="H3")
    ap.add_argument("--config-template", default="/usr/local/etc/baldr/baldr_config_{beam}.toml")
    ap.add_argument("--telem-root", default=str(Path.home() / "data" / "baldr_telem"))
    args = ap.parse_args()

    procs = []

    for b in BEAMS:
        argv = [
            "/home/you/miniconda3/envs/dcs/bin/baldr-server",
            "--beam", str(b),
            "--phasemask", args.phasemask,
            "--config", args.config_template.format(beam=b),
        ]

        launch_in_xterm(f"Baldr RTC beam {b}", argv)

    return 0
