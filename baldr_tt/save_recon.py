"""
Save an interaction matrix.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from astropy.io import fits
import time

from dcs.ZMQutils import ZmqReq

BEAM_TO_PORT = {
    1: 6673,
    2: 6674,
    3: 6675,
    4: 6676,
}

DEFAULT_BEAM = 1
DEFAULT_HOST = "mimir"
DEFAULT_OUTPUT_ROOT = "/data/custom/fdpr"

_SOCKET = None
_AMP = 0.04
_N_MODES = 11
_SETTLE_SEC = 1.0
_N_ITER = 1
_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
_RUN_TIMESTAMP = None


def get_zmq_socket(beam, host=DEFAULT_HOST):
    if beam not in BEAM_TO_PORT:
        raise ValueError(f"Invalid beam {beam}. Expected one of {sorted(BEAM_TO_PORT)}")
    port = BEAM_TO_PORT[beam]
    endpoint = f"tcp://{host}:{port}"
    print(f"Connecting to beam {beam} on {endpoint}")
    return ZmqReq(endpoint)


def get_ims(socket, cmd):
    resp = socket.send_payload(cmd, is_str=True, decode_ascii=False)
    if not isinstance(resp, dict):
        raise RuntimeError(f"No valid response for command '{cmd}': {resp}")

    width = resp["width"]
    imp = np.frombuffer(base64.b64decode(resp["im_plus_sum_encoded"]), dtype=np.float32)
    imp = imp.reshape((width, width))
    imm = np.frombuffer(
        base64.b64decode(resp["im_minus_sum_encoded"]), dtype=np.float32
    )
    imm = imm.reshape((width, width))
    return imp, imm


def run_fpdr_pokes_single(n_iter, socket=None, amp=None, n_modes=None, settle_sec=None):
    if socket is None:
        socket = _SOCKET
    if socket is None:
        raise RuntimeError("Socket not initialized. Set _SOCKET or pass socket=.")
    if amp is None:
        amp = _AMP
    if n_modes is None:
        n_modes = _N_MODES
    if settle_sec is None:
        settle_sec = _SETTLE_SEC

    all_iters = []
    for _ in range(n_iter):
        iter_ims = []
        _ = get_ims(socket, "poke 0,0")
        time.sleep(settle_sec)
        for mode in range(n_modes):
            iter_ims.append(get_ims(socket, f"poke {mode},{amp}"))
            time.sleep(settle_sec)
            iter_ims.append(get_ims(socket, f"poke {mode},{-amp}"))
            time.sleep(settle_sec)
        iter_ims.append(get_ims(socket, "poke 0,0"))
        all_iters.append(np.array(iter_ims, dtype=np.float32))

    return np.array(all_iters, dtype=np.float32)


def run_pokes_and_save(beam):
    socket = get_zmq_socket(beam, host=DEFAULT_HOST)
    ims = run_fpdr_pokes_single(
        _N_ITER,
        socket=socket,
        amp=_AMP,
        n_modes=_N_MODES,
        settle_sec=_SETTLE_SEC,
    )
    if _N_ITER == 1:
        ims = ims[0]

    beam_dir = Path(_OUTPUT_ROOT) / f"beam{beam}"
    beam_dir.mkdir(parents=True, exist_ok=True)
    output_path = beam_dir / f"{_RUN_TIMESTAMP}.fits"

    hdu = fits.PrimaryHDU(data=ims)
    hdu.writeto(output_path, overwrite=True)
    print(f"Saved beam {beam} data to {output_path}")
    return str(output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save Baldr TT interaction matrix data"
    )
    parser.add_argument(
        "--beam", type=int, required=True, choices=[-1] + sorted(BEAM_TO_PORT)
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--n-iter", type=int, default=1)
    parser.add_argument("--amp", type=float, default=0.04)
    parser.add_argument("--n-modes", type=int, default=11)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    DEFAULT_HOST = args.host
    _AMP = args.amp
    _N_MODES = args.n_modes
    _SETTLE_SEC = args.settle_sec
    _N_ITER = args.n_iter
    _OUTPUT_ROOT = args.output_root
    _RUN_TIMESTAMP = time.strftime("%Y%m%dT%H%M%S", time.gmtime())

    if args.beam == -1:
        beams = sorted(BEAM_TO_PORT)
        with ThreadPoolExecutor(max_workers=len(beams)) as executor:
            futures = {
                executor.submit(run_pokes_and_save, beam_id): beam_id
                for beam_id in beams
            }
            for future in as_completed(futures):
                beam_id = futures[future]
                try:
                    future.result()
                except (RuntimeError, ValueError, OSError) as exc:
                    print(f"Beam {beam_id} failed: {exc}")
    else:
        run_pokes_and_save(args.beam)
