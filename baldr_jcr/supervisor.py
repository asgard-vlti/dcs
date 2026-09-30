#!/usr/bin/env python

import click
import base64
import numpy as np
from astropy.io import fits  # type: ignore
import time
from typing import Dict, List, Sequence, Tuple, Optional

from numpy.typing import NDArray
from dcs.ZMQutils import ZmqReq  # type: ignore
from os import path
from dataclasses import dataclass, field
import modal_basis  # type: ignore
from enum import Enum
import pca  # type: ignore
from click import Context


# for python < 3.11 compatibility, define StrEnum here instead of importing
class StrEnum(str, Enum):
    pass


import os

# TODO: NOT REALLY SAFE: These parameters are defined both in baldr.h and here,
# I should find a way to merge these into a single source of truth.
N_MODES = 144  # TODO: change to 144, also in baldr.cpp
WIDTH = 17
N_PIXELS = WIDTH * WIDTH
SUBARRAY_WIDTH = 32
N_SUBARRAY_PIXELS = SUBARRAY_WIDTH * SUBARRAY_WIDTH
FILTER_LEN = 1
N_ACTX = 12
N_ACTUATORS = N_ACTX * N_ACTX
DIST_LEN = 10

# local constants:
BALDR_ROOT_DEFAULT = path.abspath(path.dirname(__file__))

DTYPE = np.float64
BEAM_TO_PORT = {
    1: 6662,
    2: 6663,
    3: 6664,
    4: 6665,
}

DEFAULT_HOST = os.environ.get("BALDR_HOST", default="mimir")

# Default values, will be overridden by CLI arguments
POKE: float = 0.02
ALPHA: float = 1.0
# MEAS_SCALE: float = 1 / 1000
CNT_MIN: int = 3  # minimum number of measurements to wait after applying poke
NAVG: int = 5  # number of frames to average for a poke

XC_OFFSET: float = 0.0
YC_OFFSET: float = 0.0
FLUX_MASK_RADIUS: float = 1e6 / 2.0  # use every pixel
STREHL_MASK_INNER_RADIUS: float = WIDTH / 2.0 + 1.0
STREHL_MASK_OUTER_RADIUS: float = WIDTH / 2.0 + 6.0

ARRAY_NAMES = [
    "meas_offset_0",
    "meas_offset_1",
    "flux_mask",
    "strehl_mask",
    "meas_to_mode",
    "filter_coeff_in",
    "filter_coeff_out",
    "mode_offset",
    "mode_max",
    "mode_min",
    "mode_to_com",
    "com_max",
    "com_min",
]
if DIST_LEN > 0:
    ARRAY_NAMES += ["com_dist_buffer"]

ARRAY_SHAPES = {
    "meas_offset_0": (N_PIXELS,),
    "meas_offset_1": (N_PIXELS,),
    "flux_mask": (N_SUBARRAY_PIXELS,),
    "strehl_mask": (N_SUBARRAY_PIXELS,),
    "meas_to_mode": (N_MODES, N_PIXELS),
    "filter_coeff_in": (FILTER_LEN, N_MODES),
    "filter_coeff_out": (FILTER_LEN, N_MODES),
    "mode_offset": (N_MODES,),
    "mode_max": (N_MODES,),
    "mode_min": (N_MODES,),
    "mode_to_com": (N_ACTUATORS, N_MODES),
    "com_max": (N_ACTUATORS,),
    "com_min": (N_ACTUATORS,),
    "com_dist_buffer": (N_ACTUATORS, DIST_LEN),
}

MODAL_BASIS = modal_basis.FourierModified()
# MODAL_BASIS = modal_basis.Fourier()
# MODAL_BASIS = modal_basis.Zonal()
# MODAL_BASIS = modal_basis.Zernike()

# make sure that all named arrays have an entry in this dict:
for array_name in ARRAY_NAMES:
    assert array_name in ARRAY_SHAPES.keys()

INIT_VAL = {
    "meas_offset_0": 0.0,
    "meas_offset_1": 0.0,
    "flux_mask": 1.0,
    "strehl_mask": 1.0,
    "meas_to_mode": 0.0,
    "filter_coeff_in": 0.0,
    "filter_coeff_out": 0.0,
    "mode_offset": 0.0,
    "mode_max": 1e6,
    "mode_min": -1e6,
    "mode_to_com": 0.0,
    "com_max": 0.5,
    "com_min": -0.5,
    "com_dist_buffer": 0.0,
}
# make sure that all named arrays have an entry in this dict:
for array_name in ARRAY_NAMES:
    assert array_name in INIT_VAL.keys()


class ServoMode(StrEnum):
    SERVO_OFF = "off"
    SERVO_OPEN = "open"
    SERVO_CLOSED = "closed"


class RefImType(StrEnum):
    SKY = "sky"
    LAB = "lab"


class ZmqNoResponse(RuntimeError):
    """local error type for handling an offline RTC"""

    pass


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class Beam:
    """Object for managing all interactions with the RTC at the per-beam level."""

    socket: Optional[ZmqReq] = field(init=False)
    beam_id: int
    host: str = DEFAULT_HOST
    baldr_root: str = BALDR_ROOT_DEFAULT
    verbose: int = 0

    def __post_init__(self):
        try:
            self.socket = self.get_zmq_socket()
        except:
            self.socket = None
        print(f"{bcolors.OKBLUE}BALDR_HOST={self.host}{bcolors.ENDC}")

    def get_zmq_socket(self) -> ZmqReq:
        if self.beam_id not in BEAM_TO_PORT:
            raise ValueError(
                f"Invalid beam {self.beam_id}. Expected one of {sorted(BEAM_TO_PORT)}"
            )
        port = BEAM_TO_PORT[self.beam_id]
        endpoint = f"tcp://{self.host}:{port}"
        if self.verbose:
            print(f"Connecting to beam {self.beam_id} on {endpoint}")
        return ZmqReq(endpoint)

    def request(self, message: str):
        if self.verbose > 1:
            print(f"request: {message}")
        resp = self.socket.send_payload(message, is_str=True, decode_ascii=False)  # type: ignore
        if self.verbose > 1:
            print(f"response: {resp}")
        if resp is None:
            raise ZmqNoResponse(f"No reply from RTC for command '{message}'")
        if not isinstance(resp, dict):
            raise ZmqNoResponse(f"Invalid response for command '{message}': {resp}")
        if "status_code" not in resp.keys():
            raise RuntimeError(
                f"invalid response from request.\nMessage: `{message}`\nResponse: `{resp}`"
            )
        if resp["status_code"] != 0:
            raise RuntimeError(resp["data"])
        return resp["data"]

    def get_meas(self) -> Tuple[int, np.ndarray]:
        command = "meas"
        data = self.request(command)
        meas = np.frombuffer(base64.b64decode(data["meas"]), dtype=np.float64).copy()
        return (int(data["cnt"]), meas)

    def get_mode(self) -> Tuple[int, np.ndarray]:
        command = "mode"
        data = self.request(command)
        mode = np.frombuffer(base64.b64decode(data["mode"]), dtype=np.float64).copy()
        return (int(data["cnt"]), mode)

    def avg_meas(self, *, navg: int, after_frame: int) -> np.ndarray:
        """Take `navg` frames and average them, but don't start collecting frames
        until at least `after_frames` frames have passed (e.g., to let the DM
        settle)
        """
        if navg == 0:
            raise ValueError("number of frames to average must be at least 1")
        cnt0, _ = self.get_meas()
        while True:
            cnt, meas = self.get_meas()
            if cnt >= cnt0 + after_frame:
                break
        prev_cnt = cnt
        im_avg = meas
        for i in range(navg - 1):
            while True:
                cnt, meas = self.get_meas()
                if cnt > prev_cnt:
                    im_avg += meas
                    prev_cnt = cnt
                    break
                time.sleep(1e-2)
        im_avg /= navg
        return im_avg

    @property
    def file_prefix(self) -> str:
        return path.join(self.baldr_root, f"B{self.beam_id}_")

    @staticmethod
    def check_name(name: str):
        if name not in ARRAY_NAMES:
            raise ValueError(
                f"invalid object name: {name}, must be one of: {ARRAY_NAMES}"
            )

    def writefits(self, *, name: str, array: Optional[NDArray] = None):
        self.check_name(name)
        filename = self.file_prefix + name + ".fits"
        TARGET_SHAPE = ARRAY_SHAPES[name]
        if array is None:
            array = np.ones(TARGET_SHAPE) * INIT_VAL[name]
        assert array is not None
        if array.shape != TARGET_SHAPE:
            raise IndexError(
                f"{name}.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
            )
        fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)

    def update_array(self, *, name: str, array: NDArray, push_rtc: bool = True):
        """Convenience function for updating RTC parameters from numpy arrays"""
        self.check_name(name)
        self.writefits(name=name, array=array)
        if push_rtc:
            self.request(name)

    def read_array(self, *, name: str) -> NDArray:
        """Read arrays by name from nominal data directory (BALDR_ROOT).

        E.g., imat = beam.read_array(name="mode_to_meas")
        """
        return fits.getdata(self.file_prefix + name + ".fits")  # type: ignore

    ############################################################
    ### High level functions for executing supervisory tasks ###
    ############################################################

    def reset(self, *, init: bool = False, push_rtc: bool = True):
        for name in ARRAY_NAMES:
            if init:
                self.writefits(name=name, array=None)
        if init:
            # Flux mask requires special treatment:
            self.init_flux_mask()
            self.init_strehl_mask()
        for name in ARRAY_NAMES:
            if push_rtc:
                self.request(name)
        self.request("reset")

    def init_flux_mask(self):
        xx, yy = np.meshgrid(
            np.arange(SUBARRAY_WIDTH) * 1.0,
            np.arange(SUBARRAY_WIDTH) * 1.0,
            indexing="xy",
        )
        rr = (
            (xx.flatten() - (SUBARRAY_WIDTH - 1) / 2 - XC_OFFSET) ** 2.0
            + (yy.flatten() - (SUBARRAY_WIDTH - 1) / 2 - YC_OFFSET) ** 2.0
        ) ** 0.5
        array = (rr < FLUX_MASK_RADIUS) * 1.0
        self.writefits(name="flux_mask", array=array)

    def init_strehl_mask(self):
        xx, yy = np.meshgrid(
            np.arange(SUBARRAY_WIDTH) * 1.0,
            np.arange(SUBARRAY_WIDTH) * 1.0,
            indexing="xy",
        )
        rr = (
            (xx.flatten() - (SUBARRAY_WIDTH - 1) / 2 - XC_OFFSET) ** 2.0
            + (yy.flatten() - (SUBARRAY_WIDTH - 1) / 2 - YC_OFFSET) ** 2.0
        ) ** 0.5
        array = (rr < STREHL_MASK_OUTER_RADIUS) * 1.0
        array *= rr > STREHL_MASK_INNER_RADIUS
        self.writefits(name="strehl_mask", array=array)

    def set_gain_leak(
        self,
        *,
        gain: float = 0.3,
        leak: float = 0.999,
    ):
        # compute filter coeffs from Exponentially weighted moving average gain
        filter_coeff_in = np.zeros([FILTER_LEN, N_MODES])
        filter_coeff_out = np.zeros([FILTER_LEN, N_MODES])
        filter_coeff_in[0, :] = -gain
        filter_coeff_out[0, :] = leak

        # save coeffs to fits files and push to rtc
        self.update_array(name="filter_coeff_in", array=filter_coeff_in)
        self.update_array(name="filter_coeff_out", array=filter_coeff_out)

    def flatten_offsets(self):
        array = np.zeros((N_MODES,))
        self.update_array(name="mode_offset", array=array)

    def poke(self, array: NDArray):
        self.update_array(name="mode_offset", array=array)

    def measure_interaction_matrix(
        self,
        *,
        navg: int = 5,
        poke: float = POKE,
        nmodes: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.set_servo_mode(mode=ServoMode.SERVO_OPEN)
        self.flatten_offsets()

        # we only want to poke the first `nmodes`, but if it's not specified we
        # will poke all modes (i.e., up to N_MODES)
        if nmodes is None:
            nmodes = N_MODES

        # record reference measurement
        ref_meas = self.avg_meas(navg=navg, after_frame=CNT_MIN)

        mode_to_meas = np.zeros((N_PIXELS, N_MODES), dtype=DTYPE)
        for i in range(nmodes):
            mode = np.zeros(N_MODES)
            # poke mode i (positive poke)
            print(
                f"poking mode {i}/{nmodes} for {navg} frames, {poke:0.2e}.",
                end="",
                flush=True,
            )
            mode[i] = poke
            self.poke(array=mode)
            meas_pos = self.avg_meas(navg=navg, after_frame=CNT_MIN)
            print(" POS.", end="", flush=True)
            # poke mode i (negative poke)
            mode[i] = -poke
            self.poke(array=mode)
            meas_neg = self.avg_meas(navg=navg, after_frame=CNT_MIN)
            print(f" NEG.")
            meas = (meas_pos - meas_neg) / (2 * poke)

            # inject it to matrix
            mode_to_meas[:, i] = meas
        self.flatten_offsets()
        # This matrix is manually written, since the RTC doesn't need it so it
        # doesn't enter the list of "controlled" arrays defined at the start
        # of this script.
        fits.writeto(
            self.file_prefix + "mode_to_meas.fits", mode_to_meas, overwrite=True
        )
        fig = pca.main(self.beam_id, baldr_root=self.baldr_root, plot_lim=8)
        fig.savefig(self.file_prefix + "pca.png")
        return (mode_to_meas, -ref_meas)

    def take_ref_im(self, *, ref_im_types: List[RefImType], navg: int = 5):
        if len(ref_im_types) == 0:
            raise ValueError("Must specify at least one type of reference image")
        # record reference measurement
        ref_meas = self.avg_meas(navg=navg, after_frame=CNT_MIN)
        meas_offset = -ref_meas
        for ref_im_type in ref_im_types:
            if ref_im_type.value == "sky":
                self.update_array(name=f"meas_offset_0", array=meas_offset)
            if ref_im_type.value == "lab":
                self.update_array(name=f"meas_offset_1", array=meas_offset)

    def create_modes(self):
        """Define the modal basis to be used in the RTC.

        This will create a "full" set of N_MODES modes, but not necessarily all
        of them will be used at any given time, based on imat and cmat
        computation parameters.
        """
        ### Build mode_to_com projection
        mode_to_com = MODAL_BASIS.modes_on_unit_disk(nsamplex=N_ACTX, nmodes=N_MODES)
        # filter piston from commands explicitly:
        piston_filter = np.eye(N_ACTUATORS) - 1 / N_ACTUATORS * np.ones(
            [N_ACTUATORS, N_ACTUATORS]
        )
        mode_to_com = piston_filter @ mode_to_com

        ### Measure mode_to_slope interaction
        # flatten DM
        # self.flatten_dm() # <- this sets the gain/leak to zeros, and is now not needed
        #    because we "open the loop", disabling the output of the
        #    IIR filter.
        self.flatten_offsets()
        self.reset()

        # set mode_to_com
        self.update_array(name="mode_to_com", array=mode_to_com)

    def compute_control_matrix(
        self,
        *,
        alpha: float = ALPHA,
        nmodes: Optional[int] = None,
    ):
        # measured modal imat
        mode_to_meas = self.read_array(name="mode_to_meas")

        ### Invert mode_to_slope to build slope_to_mode reconstructor
        if nmodes is None:
            nmodes = N_MODES
        meas_to_mode = np.zeros((N_MODES, N_PIXELS), dtype=DTYPE)
        meas_to_mode[:nmodes, :] = np.linalg.solve(
            mode_to_meas[:, :nmodes].T @ mode_to_meas[:, :nmodes]
            + alpha * np.eye(nmodes),
            mode_to_meas[:, :nmodes].T,
        )
        self.update_array(name="meas_to_mode", array=meas_to_mode)

    def compute_experimental_control_matrix(
        self,
        *,
        thresh: float,
        alpha: float = ALPHA,
        nmodes: Optional[int] = None,
    ):
        # measured modal imat
        mode_to_meas = self.read_array(name="mode_to_meas")
        # ideal reference intensity
        meas_offset_1 = self.read_array(name="meas_offset_1").flatten()
        ref_intensity = -meas_offset_1
        # we may want to start with a histogram of meas_offset_1
        # to determine thresh

        import matplotlib.pyplot as plt
        plt.hist(ref_intensity.flatten(), bins=100)
        plt.show()

        # set all measurements that belong to dim pixels to zero:
        mode_to_meas_filt = mode_to_meas.copy()
        mode_to_meas_filt[ref_intensity < thresh, :] = 0.0
        # import matplotlib.pyplot as plt
        # plt.plot(ref_intensity)
        # plt.show()
        fig = pca.pca(
            d=mode_to_meas,
            dtd=mode_to_meas_filt.T @ mode_to_meas_filt
            + 3e-3 * np.eye(mode_to_meas_filt.shape[1]),
            plot_lim=12,
        )
        # fig = pca.pca(
        #     d=mode_to_meas,
        #     dtd=mode_to_meas.T @ mode_to_meas,
        #     plot_lim=12,
        # )
        fig.savefig("test_pca.png")
        ### Invert mode_to_slope to build slope_to_mode reconstructor
        if nmodes is None:
            nmodes = N_MODES
        meas_to_mode = np.zeros((N_MODES, N_PIXELS), dtype=DTYPE)
        meas_to_mode[:nmodes, :] = np.linalg.solve(
            mode_to_meas_filt[:, :nmodes].T @ mode_to_meas_filt[:, :nmodes]
            + alpha * np.eye(nmodes),
            mode_to_meas_filt[:, :nmodes].T,
        )
        self.update_array(name="meas_to_mode", array=meas_to_mode)

    def set_servo_mode(self, *, mode: ServoMode):
        resp = self.request(f'servo "{mode.value}"')
        print(resp)

    def print_status(self):
        resp = self.request("status")
        print(resp)
        resp = self.request("settings")
        print(resp)

    def set_flux_thresh(self, *, thresh: float):
        resp = self.request(f"flux_threshold {thresh}")
        print(resp)

    def set_meas_offset_interp(self, *, meas_offset_interp: float):
        resp = self.request(f"meas_offset_interp {meas_offset_interp}")
        print(resp)


@click.group(help="""
This tool is a high-layer abstraction over the Baldr RTC configuration
intended to be used from the command line while the baldr RTC is running.
It connects with the RTC instance via ZMQ over a pre-defined TCP socket.
""")
@click.argument("beam", type=int, nargs=1)
@click.option("-v", "--verbose", count=True, help="set the verbosity level")
@click.pass_context
def main(ctx: Context, beam: int, verbose: int):
    if beam not in [-1, 1, 2, 3, 4]:
        raise click.BadParameter("beam must be 1, 2, 3, or 4, or -1 for all beams")
    baldr_root = os.environ.get("BALDR_ROOT")
    if baldr_root is None:
        print(
            bcolors.WARNING + "WARNING: Environment variable BALDR_ROOT not set,\n"
            f"defaulting to {BALDR_ROOT_DEFAULT}.\n"
            "Consider setting BALDR_ROOT explicitly, for example:\n"
            "    export BALDR_ROOT=/usr/local/etc" + bcolors.ENDC
        )
        baldr_root = BALDR_ROOT_DEFAULT

    if verbose > 0:
        print("VERBOSE MODE ON")

    ctx.obj = {}
    if beam == -1:
        ctx.obj["beams"] = [
            Beam(beam_id=1, baldr_root=baldr_root, verbose=verbose),
            Beam(beam_id=2, baldr_root=baldr_root, verbose=verbose),
            Beam(beam_id=3, baldr_root=baldr_root, verbose=verbose),
            Beam(beam_id=4, baldr_root=baldr_root, verbose=verbose),
        ]
    else:
        ctx.obj["beams"] = [Beam(beam_id=beam, baldr_root=baldr_root)]


@main.command(help="initialise offline RTC parameters.")
@click.pass_context
def init(ctx: Context):
    beams: List[Beam] = ctx.obj["beams"]
    print("initing!")
    try:
        [beam.reset(init=True) for beam in beams]
    except ZmqNoResponse:
        print("""
Succesfullly initialised arrays and wrote them to disk, but didn't
update them on the live RTC.

This is correct behaviour if the RTC is not yet running.
""")


if DIST_LEN > 0:

    @main.command(help="inject or disable a disturbance on the DM")
    @click.option(
        "--scale",
        type=float,
        default=1.0,
    )
    @click.option(
        "--off",
        flag_value=True,
    )
    @click.pass_context
    def disturb(ctx: Context, scale: float, off: bool):
        beams: List[Beam] = ctx.obj["beams"]
        if off:
            disturbance = np.zeros([N_ACTUATORS, DIST_LEN]) + 0.5
            [
                beam.update_array(name="com_dist_buffer", array=disturbance)
                for beam in beams
            ]
        else:
            # The default disturbance is a sine wave that sweeps accross the dm
            # over 20 frames.
            _, xx = np.meshgrid(
                np.linspace(0, 2 * 2 * np.pi, N_ACTX + 1)[:-1],
                np.linspace(0, 2 * 2 * np.pi, N_ACTX + 1)[:-1],
                indexing="ij",
            )
            xx_flat = xx.flatten()
            xx_flat -= xx_flat.mean()
            disturbance = np.zeros([N_ACTUATORS, DIST_LEN])
            for i, t in enumerate(np.linspace(0, 2 * np.pi, DIST_LEN + 1)[:-1]):
                disturbance[:, i] = 0.02 * xx_flat * scale + 0.5
            [
                beam.update_array(name="com_dist_buffer", array=disturbance)
                for beam in beams
            ]


@main.command(help="modify live RTC parameters online")
@click.option("--gain", type=float, help="set the gain (must also pass leak)")
@click.option("--leak", type=float, help="set the leak (must also pass gain)")
@click.option("--interp", type=float, help="set the interpolation parameter")
@click.option("--flux-thresh", type=float, help="set the flux threshold")
@click.option("--reset", flag_value=True, help="reset all live values in the RTC")
@click.option(
    "--open",
    "servo_mode_str",
    flag_value=ServoMode.SERVO_OPEN.value,
    help="open the loop",
)
@click.option(
    "--close",
    "servo_mode_str",
    flag_value=ServoMode.SERVO_CLOSED.value,
    help="close the loop",
)
@click.option(
    "--off",
    "servo_mode_str",
    flag_value=ServoMode.SERVO_OFF.value,
    help="stop the loop",
)
@click.pass_context
def ctrl(
    ctx: Context,
    gain: Optional[float],
    leak: Optional[float],
    interp: Optional[float],
    flux_thresh: Optional[float],
    servo_mode_str: Optional[str],
    reset: bool,
):
    beams: List[Beam] = ctx.obj["beams"]
    action_performed = False
    if gain is None and leak is not None:
        raise click.BadParameter("gain must only be set if also passing leak")
    if leak is None and gain is not None:
        raise click.BadParameter("leak must only be set if also passing gain")
    if gain is not None and leak is not None:
        for beam in beams:
            beam.set_gain_leak(gain=gain, leak=leak)
        action_performed = True

    if servo_mode_str is not None:
        servo_mode = ServoMode(servo_mode_str)
        for beam in beams:
            beam.set_servo_mode(mode=servo_mode)
        action_performed = True

    if reset:
        print("resetting!")
        [beam.reset() for beam in beams]
        action_performed = True

    if flux_thresh is not None:
        [beam.set_flux_thresh(thresh=flux_thresh) for beam in beams]
        action_performed = True

    if interp is not None:
        [beam.set_meas_offset_interp(meas_offset_interp=interp) for beam in beams]
        action_performed = True

    if not action_performed:
        print("""
WARNING: no actions were taken during the execution of this program.
This is probably unintentional. Check your command line arguments, or try with --help
""")


@main.command(help="construct an interaction matrix/perform a poke test")
@click.option("--poke", type=float, default=POKE, help="poke value to use")
@click.option("--nmodes", type=int, default=N_MODES, help="number of modes to poke")
@click.option(
    "--navg", type=int, default=NAVG, help="number of frames to average per poke"
)
@click.pass_context
def imat(ctx: Context, poke: float, nmodes: int, navg: int):
    beams: List[Beam] = ctx.obj["beams"]
    for beam in beams:
        beam.create_modes()
        beam.measure_interaction_matrix(nmodes=nmodes, poke=poke, navg=navg)


@main.command(help="build a control matrix from the interaction matrix")
@click.option(
    "--alpha",
    type=float,
    default=ALPHA,
    help="regularisation parameter for matrix inversion",
)
@click.option(
    "--nmodes",
    type=int,
    default=N_MODES,
    help="number of modes to invert, should be at most nmodes used in imat",
)
@click.pass_context
def cmat(ctx: Context, alpha: float, nmodes: int):
    beams: List[Beam] = ctx.obj["beams"]
    for beam in beams:
        beam.compute_control_matrix(nmodes=nmodes, alpha=alpha)


@main.command(help="measure a reference image for the control pipeline")
@click.option("--sky", flag_value=True, help="measuring sky reference")
@click.option("--lab", flag_value=True, help="measuring lab reference")
@click.option("--navg", type=int, default=NAVG, help="number of frames to average")
@click.pass_context
def ref(ctx: Context, sky: bool, lab: bool, navg: int):
    beams: List[Beam] = ctx.obj["beams"]
    if not any([sky, lab]):
        raise click.BadParameter("at least one of --sky or --lab must be passed")
    ref_im_types: List[RefImType] = []
    if sky:
        ref_im_types.append(RefImType.SKY)
    if lab:
        ref_im_types.append(RefImType.LAB)
    for beam in beams:
        beam.take_ref_im(ref_im_types=ref_im_types, navg=navg)


@main.command(help="probe and print the RTC status")
@click.pass_context
def status(ctx: Context):
    beams: List[Beam] = ctx.obj["beams"]
    for beam in beams:
        beam.print_status()


@main.command(help="experimental controller")
@click.option(
    "--alpha",
    type=float,
    default=ALPHA,
    help="regularisation parameter for matrix inversion",
)
@click.option(
    "--thresh",
    type=float,
    help="cutoff threshold for reference intensities to be used in cmat",
    required=True,
)
@click.option(
    "--nmodes",
    type=int,
    default=N_MODES,
    help="number of modes to invert, should be at most nmodes used in imat",
)
@click.pass_context
def cprime(ctx: Context, alpha: float, nmodes: int, thresh: float):
    beams: List[Beam] = ctx.obj["beams"]
    for beam in beams:
        beam.compute_experimental_control_matrix(thresh=thresh, nmodes=nmodes, alpha=alpha)


if __name__ == "__main__":
    try:
        main()
    except ZmqNoResponse as e:
        print(f"ZMQ Error: {e}")
        print(
            f"{bcolors.FAIL}No response from RTC server, is it running?{bcolors.ENDC}"
        )
        exit(1)
