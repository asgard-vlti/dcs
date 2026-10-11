"""
Save an interaction matrix.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import signal
import sys
import threading
import numpy as np
from astropy.io import fits
from scipy.linalg import svd
import time

from dcs.ZMQutils import ZmqReq

BEAM_TO_PORT = {
    1: 6671,
    2: 6672,
    3: 6673,
    4: 6674,
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
args=None
_CANCEL = threading.Event()


class ReconCancelled(Exception):
    pass


def _request_cancel(signum, _frame):
    if not _CANCEL.is_set():
        _CANCEL.set()
        raise ReconCancelled(f"Received signal {signum}")


def _check_cancelled():
    if _CANCEL.is_set():
        raise ReconCancelled("Reconstructor stopped")


def _wait_for_settle(seconds):
    if _CANCEL.wait(seconds):
        _check_cancelled()


def stop_and_clear_beam(beam):
    socket = get_zmq_socket(beam, host=DEFAULT_HOST)
    try:
        reply = socket.send_payload("stop_clear_dm", is_str=True, decode_ascii=False)
        if reply is not True:
            raise RuntimeError(f"Beam {beam} did not acknowledge stop_clear_dm: {reply}")
    finally:
        socket.s.close(0)


def get_zmq_socket(beam, host=DEFAULT_HOST):
    if beam not in BEAM_TO_PORT:
        raise ValueError(f"Invalid beam {beam}. Expected one of {sorted(BEAM_TO_PORT)}")
    port = BEAM_TO_PORT[beam]
    endpoint = f"tcp://{host}:{port}"
    print(f"Connecting to beam {beam} on {endpoint}")
    return ZmqReq(endpoint)


def get_ims(socket, cmd):
    _check_cancelled()
    resp = socket.send_payload(cmd, is_str=True, decode_ascii=False)
    _check_cancelled()
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
        _wait_for_settle(settle_sec)
        for mode in range(n_modes):
            iter_ims.append(get_ims(socket, f"poke {mode},{amp}"))
            _wait_for_settle(settle_sec)
            iter_ims.append(get_ims(socket, f"poke {mode},{-amp}"))
            _wait_for_settle(settle_sec)
        iter_ims.append(get_ims(socket, "poke 0,0"))
        all_iters.append(np.array(iter_ims, dtype=np.float32))

    return np.array(all_iters, dtype=np.float32)


def run_pokes_and_save(beam):
    _check_cancelled()
    socket = get_zmq_socket(beam, host=DEFAULT_HOST)
    completed = False
    try:
        result = _run_pokes_and_save(beam, socket)
        completed = True
        return result
    finally:
        try:
            if not completed:
                try:
                    stop_and_clear_beam(beam)
                    print(f"Beam {beam} servo stopped and DM channels cleared", file=sys.stderr)
                except Exception as cleanup_error:
                    print(f"Beam {beam} cleanup failed: {cleanup_error}", file=sys.stderr)
                    raise RuntimeError(f"Beam {beam} cleanup failed") from cleanup_error
        finally:
            socket.s.close(0)


def _run_pokes_and_save(beam, socket):
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
    
    #Now make the reconstructor and save it
    modes = np.zeros((_N_MODES, ims.shape[2], ims.shape[3]))
    for i in range(_N_MODES):
        flux = np.sum(ims[2*i+1, :, :, :]) + np.sum(ims[2*i+2, :, :, :])
        modes[i] = ims[2*i+1, 0, :, :] - ims[2*i+1, 1, ::-1, ::-1] \
                - ims[2*i+2, 0, :, :] + ims[2*i+2, 1, ::-1, ::-1]
        modes[i] /= flux
        modes[i] /= _AMP
        im = modes.reshape(_N_MODES, -1).T

    #Now compute the SVD of the interaction matrix. The singular values are the square roots of the eigenvalues of the interaction matrix. The number of modes that can be controlled is determined by the number of singular values that are above a certain threshold.
    U, s, Vh = svd(im, full_matrices=False)
    print("Singular values:", s)

    #Let's remove the singular values below half the mean,
    #and add this threshold to the singular values when
    # computing the inverse
    threshold = 0.5 * np.mean(s)
    sinv = 1/(s + threshold)
    sinv[s < threshold] = 0

    recon = (Vh.T * sinv) @ U.T 

    # Save the reconstructor to a fits file
    recon_path = beam_dir / f"{_RUN_TIMESTAMP}_recon.fits"
    hdu = fits.PrimaryHDU(data=recon.T)
    hdu.writeto(recon_path, overwrite=True)
    print(f"Saved beam {beam} reconstructor to {recon_path}")

    # Unless overwrite is false, save the reconstructor as the default for this beam, i.e. without timestamp.
    if args.overwrite:
        default_recon_path = beam_dir / "recon.fits"
        hdu.writeto(default_recon_path, overwrite=True)
        print(f"Saved beam {beam} reconstructor as default to {default_recon_path}")

    return str(output_path), str(recon_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save Baldr TT interaction matrix data"
    )
    parser.add_argument(
        "--beam", type=int, default=-1, choices=[-1] + sorted(BEAM_TO_PORT)
    )
    parser.add_argument("--n-iter", type=int, default=1)
    parser.add_argument("--amp", type=float, default=0.04)
    parser.add_argument("--n-modes", type=int, default=11)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--overwrite", type=bool, default=True, help="Overwrite existing default file")
    return parser.parse_args()

def main():
    global args, _AMP, _N_MODES, _N_MODES, _SETTLE_SEC, _N_ITER, _RUN_TIMESTAMP

    args = parse_args()
    _AMP = args.amp
    _N_MODES = args.n_modes
    _SETTLE_SEC = args.settle_sec
    _N_ITER = args.n_iter
    _RUN_TIMESTAMP = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    _CANCEL.clear()
    previous_sigterm = signal.signal(signal.SIGTERM, _request_cancel)
    previous_sigint = signal.signal(signal.SIGINT, _request_cancel)
    try:
        if args.beam == -1:
            beams = sorted(BEAM_TO_PORT)
            failures = []
            with ThreadPoolExecutor(max_workers=len(beams)) as executor:
                futures = {
                    executor.submit(run_pokes_and_save, beam_id): beam_id
                    for beam_id in beams
                }
                for future in as_completed(futures):
                    beam_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(beam_id)
                        print(f"Beam {beam_id} failed: {exc}", file=sys.stderr)
            return 1 if failures else 0
        run_pokes_and_save(args.beam)
        return 0
    except ReconCancelled as exc:
        print(f"Reconstructor stopped: {exc}", file=sys.stderr)
        return 143
    except Exception as exc:
        print(f"Reconstructor failed: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        
if __name__ == "__main__":
	sys.exit(main())
