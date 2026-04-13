from __future__ import annotations
import subprocess
from pathlib import Path
import shlex
import os

# to start all the Baldr RTCs
BEAMS = [1, 2, 3, 4]


def launch_in_xterm(title: str, argv: list[str], env):
    # Build the command string safely
    cmd_str = " ".join(shlex.quote(x) for x in argv)
    print(cmd_str)

    # -hold keeps terminal open after exit (important for debugging)
    subprocess.Popen([
        "xterm",
        "-T", title,
        "-hold",
        "-e", cmd_str,
    ], env=env)

def main():
    my_env = os.environ.copy()
    #Add environment variales here as needed.
    #my_env["OPENBLAS_NUM_THREADS"] = "1"


    for b in BEAMS:
        argv = [
            "/usr/local/bin/baldr_tt",
            f"/usr/local/etc/def{b}.toml",
            "--socket", f"tcp://*:667{b}"
        ]

        launch_in_xterm(f"Baldr TT beam {b}", argv, my_env)

    return 0
    
if __name__ == "__main__":
    main()
