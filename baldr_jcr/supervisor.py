import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from astropy.io import fits  # type: ignore
import time
from typing import Tuple, Optional
from tqdm import tqdm  # type: ignore
from dcs.ZMQutils import ZmqReq  # type: ignore
from os import path

# TODO: NOT REALLY SAFE: These parameters are defined both in baldr.h and here,
# I should find a way to merge these into a single source of truth.
N_MODES = 11
WIDTH = 15
N_PIXELS = WIDTH * WIDTH
FILTER_LEN = 2
N_ACTUATORS = 144
DIST_LEN = 10

BEAM_TO_PORT = {
    1: 17474,  # 6671
    2: 6672,
    3: 6673,
    4: 6671,  # <- should this be 6674?
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


class Mode:
    pass


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
    print(resp)
    if resp["status_code"] != 0:
        raise RuntimeError(resp["data"])
    return resp["data"]

def get_meas(socket: ZmqReq) -> np.ndarray:
    command = f"meas"
    data = request(socket, command)
    meas = np.frombuffer(base64.b64decode(data["meas"]), dtype=np.float64)
    return meas


def writefits_meas_offset(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "meas_offset.fits")
):
    TARGET_SHAPE = (N_PIXELS,)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"meas_offset.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array, overwrite=True)


def writefits_meas_to_mode(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "meas_to_mode.fits")
):
    TARGET_SHAPE = (N_MODES, N_PIXELS)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"meas_to_mode.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array, overwrite=True)


def writefits_filter_coeff_in(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "filter_coeff_in.fits")
):
    TARGET_SHAPE = (N_MODES, FILTER_LEN)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"filter_coeff_in.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    print("baldr root: ", BALDR_ROOT)
    fits.writeto(filename=filename, data=array, overwrite=True)


def writefits_filter_coeff_out(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "filter_coeff_out.fits")
):
    TARGET_SHAPE = (N_MODES, FILTER_LEN)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"filter_coeff_out.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array, overwrite=True)


def writefits_mode_to_com(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "mode_to_com.fits")
):
    TARGET_SHAPE = (N_ACTUATORS, N_MODES)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"mode_to_com.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array, overwrite=True)


def writefits_com_offset(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "com_offset.fits")
):
    TARGET_SHAPE = (N_ACTUATORS,)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"com_offset.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array, overwrite=True)


def writefits_com_dist_buffer(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "com_dist_buffer.fits")
):
    TARGET_SHAPE = (N_ACTUATORS, DIST_LEN)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"com_dist_buffer.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array, overwrite=True)


def writefits_com_to_meas(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "com_to_meas.fits")
):
    TARGET_SHAPE = (N_PIXELS, N_ACTUATORS)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"com_to_meas.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array, overwrite=True)


def writefits_mode_to_meas(
    array: np.ndarray, filename=path.join(BALDR_ROOT, "mode_to_meas.fits")
):
    TARGET_SHAPE = (N_PIXELS, N_MODES)
    if array.shape != TARGET_SHAPE:
        raise IndexError(
            f"mode_to_meas.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
        )
    fits.writeto(filename=filename, data=array, overwrite=True)


def update_meas_offset(socket: ZmqReq, send: bool = True):
    """Save current image as reference measurement and load"""
    # read image via commander
    meas = get_meas(socket=socket)

    # save image to fits file
    writefits_meas_offset(meas)

    # tell commander to load image as meas ref
    if send:
        request(socket, "meas_offset")


def update_meas_to_mode(measurement: Measurement, mode: Mode, send: bool = True):
    """Build a reconstructor"""
    # build forward matrix from modes to measurements
    # TODO

    # invert forward matrix to get reconstructor
    # TODO

    # save reconstructor to default path
    # TODO

    # tell commander to load reconstructor as meas_to_mode
    # TODO

    pass


def update_filter_coeff(
    ewma_gain: float = 0.3,
    send: bool = True,  # we usually will tell commander to update the controller
):
    # compute filter coeffs from Exponentially weighted moving average gain
    filter_coeff_in = np.zeros([N_MODES, FILTER_LEN])
    filter_coeff_out = np.zeros([N_MODES, FILTER_LEN])
    filter_coeff_in[:, 0] = ewma_gain
    filter_coeff_out[:, 0] = 1 - ewma_gain

    # save coeffs to fits files
    writefits_filter_coeff_in(filter_coeff_in)
    writefits_filter_coeff_out(filter_coeff_out)

    if send:
        # tell commander to load filter coeffs
        # TODO
        pass


def update_mode_to_com(
    mode: Mode, influence_functions: InfluenceFunctions, send: bool = True
):
    """Update the modal projection matrix (from modes to DM command) using
    some global modal definition and some global influence function definition.
    """
    # define some intermediate phase space
    # TODO

    # project modes to that phase space
    # TODO

    # project commands to that phase space
    # TODO

    # linear algebra trickery to build mode to command projector
    # TODO

    pass


def update_mode(
    mode: Mode,
    influence_functions: InfluenceFunctions,
    measurement: Measurement,
    send: bool = True,
):
    """Update the defined modes and propagate to all dependencies"""
    # update mode to com
    update_mode_to_com(
        mode=mode, influence_functions=influence_functions, send=False
    )

    # update mode to com
    update_mode_to_com(
        mode=mode, influence_functions=influence_functions, send=False
    )

    # update meas to mode
    update_meas_to_mode(measurement=measurement, mode=mode, send=False)

    if send:
        # send commander message to update all relevant matrices and reset controller
        # TODO
        pass


def update_com_offset(
    array: np.ndarray,
    send: bool = True,
):
    """Update the command offset from a numpy array"""
    writefits_com_offset(array)

    if send:
        # tell commander to load com offset
        # TODO
        pass


def update_com_dist_buffer(
    array: np.ndarray,
    send: bool = True,
):
    """Update the command disturbance buffer from a numpy array"""
    writefits_com_dist_buffer(array)

    if send:
        # tell commander to load com dist buffer
        # TODO
        pass


def update_com_to_meas(
    influence_functions: InfluenceFunctions,
    measurement: Measurement,
    send: bool = True,
):
    """Build an interaction matrix from an influence function model"""
    # measure interaction matrix via commander

    # fit geometric parameters to build synthetic interaction matrix
    # (this might involve )

    # write synthetic interaction matrix to fits

    if send:
        # tell commander to load com_to_meas from file
        # TODO
        pass


if __name__ == "__main__":
    socket = get_zmq_socket(beam=1, host=DEFAULT_HOST)
    update_meas_offset(socket=socket, send=True)
    
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