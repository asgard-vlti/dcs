import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from astropy.io import fits  # type: ignore
import time
from typing import Tuple, Optional

from numpy.typing import NDArray
from dcs.ZMQutils import ZmqReq  # type: ignore
from os import path
from dataclasses import dataclass
import modal_basis

# import matplotlib.pyplot as plt

# TODO: NOT REALLY SAFE: These parameters are defined both in baldr.h and here,
# I should find a way to merge these into a single source of truth.
N_MODES = 100
WIDTH = 15
N_PIXELS = WIDTH * WIDTH
FILTER_LEN = 1
N_ACTX = 12
N_ACTUATORS = N_ACTX * N_ACTX
DIST_LEN = 100

# Default values, may be overridden by CLI arguments
POKE: float = 0.1
ALPHA: float = 500.0
BETA: float = 0.0
MEAS_SCALE: float = 1 / 1000
CNT_MIN: int = 3  # minimum number of measurements to wait after applying poke

BEAM_TO_PORT = {
    1: 6662,
    2: 6663,
    3: 6664,
    4: 6665,  
}

DEFAULT_BEAM = 1
# DEFAULT_HOST = "mimir"
DEFAULT_HOST = "localhost"
DEFAULT_OUTPUT_ROOT = "/tmp"

_SOCKET = None
_AMP = 0.04
_N_MODES = 11
_SETTLE_SEC = 1.0
_N_ITER = 1
_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
_RUN_TIMESTAMP = None

BALDR_ROOT = path.abspath(path.dirname(__file__))

DTYPE = np.float64


class InfluenceFunctions:
    pass


class Measurement:
    pass


def get_zmq_socket(beam: int, host: str = DEFAULT_HOST) -> ZmqReq:
    if beam not in BEAM_TO_PORT:
        raise ValueError(f"Invalid beam {beam}. Expected one of {sorted(BEAM_TO_PORT)}")
    port = BEAM_TO_PORT[beam]
    endpoint = f"tcp://{host}:{port}"
    print(f"Connecting to beam {beam} on {endpoint}")
    return ZmqReq(endpoint)


def request(socket: ZmqReq, message: str):
    resp = socket.send_payload(message, is_str=True, decode_ascii=False)  # type: ignore
    if not isinstance(resp, dict):
        raise RuntimeError(f"No valid response for command '{message}': {resp}")
    # print(resp)
    if resp["status_code"] != 0:
        raise RuntimeError(resp["data"])
    return resp["data"]


def get_meas(socket: ZmqReq) -> Tuple[int, np.ndarray]:
    command = f"meas"
    data = request(socket, command)
    meas = np.frombuffer(base64.b64decode(data["meas"]), dtype=np.float64).copy()
    return (int(data["cnt"]), meas)


def get_mode(socket: ZmqReq) -> Tuple[int, np.ndarray]:
    command = f"mode"
    data = request(socket, command)
    mode = np.frombuffer(base64.b64decode(data["mode"]), dtype=np.float64).copy()
    return (int(data["cnt"]), mode)


def avg_meas(socket: ZmqReq, navg: int, after_frame: int) -> np.ndarray:
    """Take `navg` frames and average them, but don't start collecting frames
    until at least `after_frames` frames have passed (e.g., to let the DM
    settle)
    """
    if navg == 0:
        raise ValueError("number of frames to average must be at least 1")
    cnt0, _ = get_meas(socket)
    while True:
        cnt, meas = get_meas(socket)
        if cnt >= cnt0 + after_frame:
            break
    prev_cnt = cnt
    im_avg = meas
    for i in range(navg - 1):
        while True:
            cnt, meas = get_meas(socket)
            if cnt > prev_cnt:
                im_avg += meas
                prev_cnt = cnt
                break
            time.sleep(1e-2)
    im_avg /= navg
    return im_avg


def fitsread(filename: str) -> np.ndarray:
    return fits.open(filename)[0].data  # type: ignore


####################################################################
### Functions for writing fits files for data ojects from arrays ###
####################################################################
# If no array is provided, then an appropriate default is set. This is
# often an array of zeros, but in the following cases it's not (for obvious
# reasons):
#  - mode_max => +1e6
#  - mode_min => -1e6
#  - com_max => +1e6
#  - com_min => -1e6


def writefits_meas_offset(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "meas_offset.fits"),
):
    TARGET_SHAPE = (N_PIXELS,)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"meas_offset.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_meas_to_mode(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "meas_to_mode.fits"),
):
    TARGET_SHAPE = (N_MODES, N_PIXELS)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"meas_to_mode.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_filter_coeff_in(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "filter_coeff_in.fits"),
):
    TARGET_SHAPE = (FILTER_LEN, N_MODES)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"filter_coeff_in.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    print("baldr root: ", BALDR_ROOT)
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_filter_coeff_out(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "filter_coeff_out.fits"),
):
    TARGET_SHAPE = (FILTER_LEN, N_MODES)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"filter_coeff_out.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_mode_offset(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "mode_offset.fits"),
):
    TARGET_SHAPE = (N_MODES,)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"mode_offset.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_mode_max(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "mode_max.fits"),
):
    TARGET_SHAPE = (N_MODES,)
    if array is None:
        array = np.ones(TARGET_SHAPE) * 1e6
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"mode_max.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_mode_min(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "mode_min.fits"),
):
    TARGET_SHAPE = (N_MODES,)
    if array is None:
        array = np.ones(TARGET_SHAPE) * -1e6
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"mode_min.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_mode_to_com(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "mode_to_com.fits"),
):
    TARGET_SHAPE = (N_ACTUATORS, N_MODES)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"mode_to_com.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_com_max(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "com_max.fits"),
):
    TARGET_SHAPE = (N_ACTUATORS,)
    if array is None:
        array = np.ones(TARGET_SHAPE) * 1e6
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"com_max.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_com_min(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "com_min.fits"),
):
    TARGET_SHAPE = (N_ACTUATORS,)
    if array is None:
        array = np.ones(TARGET_SHAPE) * -1e6
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"com_min.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_com_dist_buffer(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "com_dist_buffer.fits"),
):
    TARGET_SHAPE = (N_ACTUATORS, DIST_LEN)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"com_dist_buffer.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_com_offset(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "com_offset.fits"),
):
    TARGET_SHAPE = (N_ACTUATORS,)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"com_offset.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


def writefits_com_to_meas(
    array: Optional[np.ndarray] = None,
    filename=path.join(BALDR_ROOT, "com_to_meas.fits"),
):
    TARGET_SHAPE = (N_PIXELS, N_ACTUATORS)
    if array is None:
        array = np.zeros(TARGET_SHAPE)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"com_to_meas.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)


##########################################################################
### Convenience function for updating RTC parameters from numpy arrays ###
##########################################################################


def update_meas_offset(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the meas_offset from a numpy array"""
    writefits_meas_offset(array)
    if socket is not None:
        request(socket, "meas_offset")


def update_meas_to_mode(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the meas_to_mode from a numpy array"""
    writefits_meas_to_mode(array)
    if socket is not None:
        request(socket, "meas_to_mode")


def update_filter_coeff_in(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the filter_coeff_in from a numpy array"""
    writefits_filter_coeff_in(array)
    if socket is not None:
        request(socket, "filter_coeff_in")


def update_filter_coeff_out(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the filter_coeff_out from a numpy array"""
    writefits_filter_coeff_out(array)
    if socket is not None:
        request(socket, "filter_coeff_out")


def update_mode_offset(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the _mode_offset from a numpy array"""
    writefits_mode_offset(array)
    if socket is not None:
        request(socket, "mode_offset")


def update_mode_max(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the mode_max from a numpy array"""
    writefits_mode_max(array)
    if socket is not None:
        request(socket, "mode_max")


def update_mode_min(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the mode_min from a numpy array"""
    writefits_mode_min(array)
    if socket is not None:
        request(socket, "mode_min")


def update_mode_to_com(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the mode_to_com from a numpy array"""
    writefits_mode_to_com(array)
    if socket is not None:
        request(socket, "mode_to_com")


def update_com_max(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the com_max from a numpy array"""
    writefits_com_max(array)
    if socket is not None:
        request(socket, "com_max")


def update_com_min(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the com_min from a numpy array"""
    writefits_com_min(array)
    if socket is not None:
        request(socket, "com_min")


def update_com_dist_buffer(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the com_dist_buffer from a numpy array"""
    writefits_com_dist_buffer(array)
    if socket is not None:
        request(socket, "com_dist_buffer")


def update_com_offset(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the com_offset from a numpy array"""
    writefits_com_offset(array)
    if socket is not None:
        request(socket, "com_offset")


def update_com_to_meas(array: np.ndarray, socket: Optional[ZmqReq] = None):
    """Update the com_to_meas from a numpy array"""
    writefits_com_to_meas(array)
    if socket is not None:
        request(socket, "com_to_meas")


def update_delay(delay: float, socket: Optional[ZmqReq] = None):
    """Update the delay from a float"""
    if socket is not None:
        request(socket, f"delay {delay}")


############################################################
### High level functions for executing supervisory tasks ###
############################################################
# Most of these functions include a "push" argument, which
# indicates that the computed variables should be immediately
# reloaded in the RTC. If push==False, then the values will
# still be saved and computed, but they won't be applied
# immediately to the RTC. The RTC initialises with whatever
# is saved on disk, so setting push==False will lead to the
# computed values being loaded next time the RTC is started.


def reset(socket: ZmqReq, *, init: bool = False, offline=False):
    if init:
        writefits_meas_offset()
        writefits_meas_to_mode()
        writefits_filter_coeff_in()
        writefits_filter_coeff_out()
        writefits_mode_offset()
        writefits_mode_max()
        writefits_mode_min()
        writefits_mode_to_com()
        writefits_com_max()
        writefits_com_min()
        writefits_com_dist_buffer()
        writefits_com_offset()
        writefits_com_to_meas()
    if not offline:
        request(socket, "meas_offset")
        request(socket, "meas_to_mode")
        request(socket, "filter_coeff_in")
        request(socket, "filter_coeff_out")
        request(socket, "mode_offset")
        request(socket, "mode_max")
        request(socket, "mode_min")
        request(socket, "mode_to_com")
        request(socket, "com_max")
        request(socket, "com_min")
        request(socket, "com_dist_buffer")
        request(socket, "com_offset")
        request(socket, "com_to_meas")


def set_leaky_gain_leak(
    socket: Optional[ZmqReq] = None,
    *,
    gain: float = 0.3,
    leak: float = 0.999,
):
    # compute filter coeffs from Exponentially weighted moving average gain
    filter_coeff_in = np.zeros([FILTER_LEN, N_MODES])
    filter_coeff_out = np.zeros([FILTER_LEN, N_MODES])
    filter_coeff_in[0, :] = -gain
    filter_coeff_out[0, :] = leak

    # save coeffs to fits files
    update_filter_coeff_in(filter_coeff_in, socket)
    update_filter_coeff_out(filter_coeff_out, socket)


def set_polc_gain(
    socket: Optional[ZmqReq] = None,
    *,
    ewma_gain: float = 0.3,
    ewma_leak: float = 1.0,
):
    # compute filter coeffs from Exponentially weighted moving average gain
    filter_coeff_in = np.zeros([FILTER_LEN, N_MODES])
    filter_coeff_out = np.zeros([FILTER_LEN, N_MODES])
    filter_coeff_in[0, :] = -ewma_gain
    filter_coeff_out[0, :] = 1 - ewma_gain
    filter_coeff_out *= ewma_leak

    # save coeffs to fits files
    update_filter_coeff_in(filter_coeff_in, socket)
    update_filter_coeff_out(filter_coeff_out, socket)


def flatten_dm(socket: ZmqReq):
    # compute filter coeffs. Zeros correspond to an "all stop" filter.
    filter_coeff_in = np.zeros([FILTER_LEN, N_MODES])
    filter_coeff_out = np.zeros([FILTER_LEN, N_MODES])
    # save coeffs to fits files
    update_filter_coeff_in(filter_coeff_in, socket)
    update_filter_coeff_out(filter_coeff_out, socket)


def set_com_clip(socket: ZmqReq, *, clip_val: float):
    # compute filter coeffs. Zeros correspond to an "all stop" filter.
    com_max = clip_val * np.ones([N_ACTUATORS])
    # save coeffs to fits files
    update_com_max(com_max, socket)
    update_com_min(-com_max, socket)


def flatten_offsets(socket: ZmqReq):
    array = np.zeros((N_MODES,))
    update_mode_offset(array, socket=socket)


def measure_interaction_matrix(
    socket: ZmqReq,
    *,
    navg: int = 5,
    poke: float = POKE,
    nmodes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    flatten_offsets(socket)

    # we only want to poke the first `nmodes`, but if it's not specified we
    # will poke all modes (i.e., up to N_MODES)
    if nmodes is None:
        nmodes = N_MODES

    # record reference measurement
    ref_meas = avg_meas(socket, navg, CNT_MIN)

    # plt.matshow(ref_meas.reshape([WIDTH, WIDTH]))
    # plt.title(f"reference image")
    # plt.colorbar()
    # plt.savefig("tmp.png")
    # plt.close()
    # the only way to get here is if ref_meas exists

    mode_to_meas = np.zeros((N_PIXELS, N_MODES), dtype=DTYPE)
    for i in range(nmodes):
        mode = np.zeros(N_MODES)
        # poke mode i (positive poke)
        mode[i] = poke
        update_mode_offset(mode, socket=socket)
        meas_pos = avg_meas(socket, navg, CNT_MIN)
        # poke mode i (negative poke)
        mode[i] = -poke
        update_mode_offset(mode, socket=socket)
        meas_neg = avg_meas(socket, navg, CNT_MIN)
        meas = (meas_pos - meas_neg) / (2 * poke) * MEAS_SCALE
        # plt.matshow(meas.reshape([WIDTH, WIDTH]))
        # plt.title(f"response to mode {i}")
        # plt.colorbar()
        # plt.savefig("tmp.png")
        # plt.close()

        # inject it to matrix
        mode_to_meas[:, i] = meas
    flatten_offsets(socket)
    return (mode_to_meas, -ref_meas)


def build_meas_to_mode(
    mode_to_meas: NDArray,
    *,
    alpha: float,
    nmodes: Optional[int],
):
    """Build the POL control matrix from the interaction matrix.

    NOTE: mode_to_meas and meas_to_mode should ALWAYS have the full N_MODES in
    the modal dimension. The nmodes argument here specifies that the matrices
    should only consume/produce up to the nmodes'th mode, and that the remainder
    of the entries in those matrices should be zero.
    """
    if nmodes is None:
        nmodes = N_MODES
    meas_to_mode = np.zeros((N_MODES, N_PIXELS), dtype=DTYPE)
    meas_to_mode[:nmodes, :] = (
        np.linalg.solve(
            mode_to_meas[:, :nmodes].T @ mode_to_meas[:, :nmodes]
            + alpha * np.eye(nmodes),
            mode_to_meas[:, :nmodes].T,
        )
        * MEAS_SCALE
    )
    return meas_to_mode


def create_polc_matrices(
    socket: ZmqReq,
    *,
    navg: int = 5,
    alpha: float = ALPHA,
    beta: float = BETA,
    poke: float = POKE,
    nmodes: Optional[int] = None,
):
    """Measure the interaction matrix, fit the system parameters based on that
    measurement, and then compute and upload all control matrices derived from
    the system parameters.

    NOTE: To avoid recompiling the RTC, the nmodes option only modifies the
    "values" of the control matrices, not the dimensions. For this reason, the
    code will sometimes refer to N_MODES (a constant) and nmodes (a variable).
    """

    ### Build mode_to_com projection
    mode_to_com = modal_basis.Zernike().modes_on_unit_disk(
        nsamplex=N_ACTX, nmodes=N_MODES
    )

    ### Measure mode_to_slope interaction
    # flatten DM
    flatten_dm(socket=socket)
    flatten_offsets(socket=socket)

    # set mode_to_com
    update_mode_to_com(mode_to_com, socket=socket)

    # measure modal imat
    mode_to_meas, meas_offset = measure_interaction_matrix(
        socket,
        navg=navg,
        poke=poke,
        nmodes=nmodes,
    )

    # run some statistics on the measured interaction matrix
    # TODO

    # produce com imat
    com_to_meas = (
        mode_to_meas
        @ np.linalg.solve(
            mode_to_com.T @ mode_to_com + beta * np.eye(mode_to_com.shape[1]),
            mode_to_com.T,
        )
        / MEAS_SCALE
    )
    update_com_to_meas(com_to_meas)

    ### Invert mode_to_slope to build slope_to_mode reconstructor
    meas_to_mode = build_meas_to_mode(mode_to_meas, alpha=alpha, nmodes=nmodes)
    update_meas_to_mode(meas_to_mode, socket=socket)
    update_meas_offset(meas_offset, socket=socket)


def create_leaky_matrices(
    socket: ZmqReq,
    *,
    navg: int = 5,
    alpha: float = ALPHA,
    poke: float = POKE,
    nmodes: Optional[int] = None,
):
    """Measure the interaction matrix, fit the system parameters based on that
    measurement, and then compute and upload all control matrices derived from
    the system parameters.

    NOTE: To avoid recompiling the RTC, the nmodes option only modifies the
    "values" of the control matrices, not the dimensions. For this reason, the
    code will sometimes refer to N_MODES (a constant) and nmodes (a variable).
    """
    ### Build mode_to_com projection
    mode_to_com = modal_basis.Zernike().modes_on_unit_disk(
        nsamplex=N_ACTX, nmodes=N_MODES
    )

    ### Measure mode_to_slope interaction
    # flatten DM
    flatten_dm(socket=socket)
    flatten_offsets(socket=socket)

    # set mode_to_com
    update_mode_to_com(mode_to_com, socket=socket)

    # measure modal imat
    mode_to_meas, meas_offset = measure_interaction_matrix(
        socket,
        navg=navg,
        poke=poke,
        nmodes=nmodes,
    )

    update_com_to_meas(np.zeros((N_PIXELS, N_ACTUATORS)), socket=socket)

    ### Invert mode_to_slope to build slope_to_mode reconstructor
    meas_to_mode = build_meas_to_mode(mode_to_meas, alpha=alpha, nmodes=nmodes)

    update_meas_to_mode(meas_to_mode, socket=socket)
    update_meas_offset(meas_offset, socket=socket)


def mode_telemetry(nframes: int, after_frame: int, socket: ZmqReq) -> np.ndarray:
    if nframes == 0:
        raise ValueError("number of frames to average must be at least 1")
    cnt0, _ = get_mode(socket)
    while True:
        cnt, mode = get_mode(socket)
        if cnt >= cnt0 + after_frame:
            break
    prev_cnt = cnt
    mode_telem = [mode]
    # note that the frames aren't guaranteed (or even expected) to be
    # contiguous in time-samples
    for i in range(nframes + 1):
        while True:
            cnt, mode = get_mode(socket)
            if cnt > prev_cnt:
                mode_telem += [mode]
                prev_cnt = cnt
                break
            time.sleep(1e-2)
    return np.array(mode_telem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Baldr Supervisor")
    parser.add_argument(
        "beam", type=int, help="index of beam [1-4]", choices=[1, 2, 3, 4]
    )
    parser.add_argument(
        "--init",
        "-i",
        help="initialise all arrays with zeros and save them to disk",
        action="count",
    )
    parser.add_argument(
        "--disturboff",
        help="resets the disturbance to zero",
        action="count",
    )
    parser.add_argument(
        "--disturb",
        help="inject a test signal onto the dms",
        action="count",
    )

    parser.add_argument(
        "--offset",
        help="inject a modal offset onto the DMs (for NCPAs later)",
        action="count",
    )

    parser.add_argument(
        "--polc",
        help="measure imat from live system, then initialise controller in POLC mode",
        action="count",
    )

    parser.add_argument(
        "--leaky",
        help="measure imat from live system, then initialise controller in leaky integrator mode",
        action="count",
    )

    parser.add_argument(
        "--gain", help="gain, only used if also used with --leaky or --polc", type=float
    )
    parser.add_argument(
        "--leak", help="leak, only used if also used with --leaky", type=float
    )
    parser.add_argument(
        "--recompute", help="recompute the specified controller", action="count"
    )

    parser.add_argument("--clipcom", help="value to clip commands to", type=float)

    parser.add_argument(
        "--poke", help="value to poke each mode, try 0.01", type=float, default=0.1
    )

    parser.add_argument("--nmodes", help="maximum mode index to control", type=int)

    args = parser.parse_args()

    action_performed = False

    if args.init is not None:
        print("initing!")
        socket = get_zmq_socket(beam=args.beam, host=DEFAULT_HOST)
        action_performed = True
        try:
            reset(socket, init=True)
        except RuntimeError:
            print("""
Succesfullly initialised arrays and wrote them to disk, but didn't
update them on the live RTC.

This is correct behaviour if the RTC is not yet running.
""")

    if args.clipcom is not None:
        socket = get_zmq_socket(beam=args.beam, host=DEFAULT_HOST)
        set_com_clip(socket, clip_val=args.clipcom)
        action_performed = True


    # TODO: Change this pipeline to only create one socket (but only crash if the
    # client needs to be online)

    if args.disturb is not None:
        if args.disturboff is not None:
            raise ValueError("cannot simultaneously be disturbing and not disturbing")
        # The default disturbance is a sine wave that sweeps accross the dm
        # over 20 frames.
        socket = get_zmq_socket(beam=args.beam, host=DEFAULT_HOST)
        _, xx = np.meshgrid(
            np.linspace(0, 2 * 2 * np.pi, N_ACTX + 1)[:-1],
            np.linspace(0, 2 * 2 * np.pi, N_ACTX + 1)[:-1],
            indexing="ij",
        )
        xx_flat = xx.flatten()
        disturbance = np.zeros([N_ACTUATORS, DIST_LEN])
        for i, t in enumerate(np.linspace(0, 2 * np.pi, DIST_LEN + 1)[:-1]):
            disturbance[:, i] = 0.1 * np.sin(xx_flat + t)
        update_com_dist_buffer(disturbance, socket=socket)
        action_performed = True

    if args.disturboff is not None:
        # The default disturbance is a sine wave that sweeps accross the dm
        # over 20 frames.
        socket = get_zmq_socket(beam=args.beam, host=DEFAULT_HOST)
        disturbance = np.zeros([N_ACTUATORS, DIST_LEN])
        update_com_dist_buffer(disturbance, socket=socket)
        action_performed = True

    if args.polc is not None:
        socket = get_zmq_socket(beam=args.beam, host=DEFAULT_HOST)
        if args.recompute is not None:
            create_polc_matrices(socket, nmodes=args.nmodes, poke=args.poke)
        if args.gain is not None:
            gain = args.gain
        else:
            gain = 0.3
        if args.leak is not None:
            leak = args.leak
        else:
            leak = 1.0
        set_polc_gain(socket, ewma_gain=gain, ewma_leak=leak)
        action_performed = True
    elif args.leaky is not None:
        socket = get_zmq_socket(beam=args.beam, host=DEFAULT_HOST)
        if args.recompute is not None:
            create_leaky_matrices(socket, nmodes=args.nmodes, poke=args.poke)
        if args.gain is not None:
            gain = args.gain
        else:
            gain = 0.3
        if args.leak is not None:
            leak = args.leak
        else:
            leak = 0.95
        set_leaky_gain_leak(socket, gain=gain, leak=leak)
        action_performed = True
    else:
        if args.gain is not None:
            raise ValueError("gain must only be set if also passing --polc or --leaky")
        if args.leak is not None:
            raise ValueError("leak must only be set if also passing --leaky")

    if not action_performed:
        print("""
WARNING: no actions were taken during the execution of this program.
This is probably unintentional. Check your command line arguments!
""")
    # socket = get_zmq_socket(beam=args.beam, host=DEFAULT_HOST)
    # create_meas_offset(socket=socket, push=True)

    # update_filter_coeff(0.3, send=False)
    # mode = Mode()
    # measurement = Measurement()
    # influence_functions = InfluenceFunctions()
    # update_mode(
    #     mode=mode,
    #     influence_functions=influence_functions,
    #     measurement=measurement,
    #     send=True,
    # )
