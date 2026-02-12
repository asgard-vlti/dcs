from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
# to start all the Baldr RTCs
BEAMS = [1, 2, 3, 4]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phasemask", default="H3")
    ap.add_argument("--config-template", default="/usr/local/etc/baldr/baldr_config_{beam}.toml")
    ap.add_argument("--telem-root", default=str(Path.home() / "data" / "baldr_telem"))
    args = ap.parse_args()

    procs = []
    for b in BEAMS:
        telem_dir = Path(args.telem_root) / f"beam{b}"
        telem_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "baldr-server",
            "--beam", str(b),
            "--phasemask", args.phasemask,
            "--config", args.config_template.format(beam=b),
            "--telem-dir", str(telem_dir),
        ]
        # detached; each beam is its own process
        procs.append(subprocess.Popen(cmd))

    return 0
