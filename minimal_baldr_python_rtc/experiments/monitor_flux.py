"""Print the rolling total flux from a Baldr camera stream.

Usage: python monitor_flux.py /dev/shm/baldr1.im.shm
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as onp


WINDOW = 1000
PRINT_INTERVAL_S = 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", help="Baldr shared-memory stream path")
    args = parser.parse_args()

    if not Path(args.stream).is_file():
        parser.error(f"stream does not exist: {args.stream}")

    from xaosim.shmlib import shm

    camera = shm(args.stream, nosem=True)
    fluxes = deque()
    total = 0.0
    counter = camera.get_counter()
    next_print = time.monotonic()

    try:
        while True:
            frame = camera.get_data(check=counter)
            frame_counter = camera.get_counter()
            if frame_counter == counter:
                continue
            if frame_counter < counter:
                raise RuntimeError("Camera counter reset; restart the monitor")
            frame = frame.copy()
            if camera.get_counter() != frame_counter:
                continue
            if frame.ndim != 2:
                raise ValueError(f"Expected a 2D Baldr frame, got {frame.shape}")
            counter = frame_counter
            flux = float(onp.sum(frame, dtype=onp.float64))

            if len(fluxes) == WINDOW:
                total -= fluxes.popleft()
            fluxes.append(flux)
            total += flux

            now = time.monotonic()
            if len(fluxes) == WINDOW and now >= next_print:
                print(f"Mean total flux (last {WINDOW} frames): {total / WINDOW:.3f}")
                next_print = now + PRINT_INTERVAL_S
    except KeyboardInterrupt:
        pass
    finally:
        camera.close(erase_file=False)


if __name__ == "__main__":
    main()
