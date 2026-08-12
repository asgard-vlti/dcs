import argparse
import base64
import numpy as np
from astropy.io import fits  # type: ignore
import time
from typing import Tuple, Optional

from numpy.typing import NDArray
from dcs.ZMQutils import ZmqReq  # type: ignore
from os import path
from dataclasses import dataclass, field
import modal_basis

# TODO: NOT REALLY SAFE: These parameters are defined both in baldr.h and here,
# I should find a way to merge these into a single source of truth.
N_MODES = 100  # TODO: change to 144, also in baldr.cpp
WIDTH = 15
N_PIXELS = WIDTH * WIDTH
FILTER_LEN = 1
N_ACTX = 12
N_ACTUATORS = N_ACTX * N_ACTX
DIST_LEN = 100

# local constants:
BALDR_ROOT = path.abspath(path.dirname(__file__))
DTYPE = np.float64
BEAM_TO_PORT = {
    1: 6662,
    2: 6663,
    3: 6664,
    4: 6665,
}
# DEFAULT_HOST = "mimir"
DEFAULT_HOST = "localhost"

# Default values, will be overridden by CLI arguments
POKE: float = 0.1
ALPHA: float = 500.0
BETA: float = 0.0
MEAS_SCALE: float = 1 / 1000
CNT_MIN: int = 3  # minimum number of measurements to wait after applying poke

ARRAY_NAMES = [
    "meas_offset",
    "meas_to_mode",
    "filter_coeff_in",
    "filter_coeff_out",
    "mode_offset",
    "mode_max",
    "mode_min",
    "mode_to_com",
    "com_max",
    "com_min",
    "com_dist_buffer",
    "com_offset",
    "com_to_meas",
]

ARRAY_SHAPES = {
    "meas_offset": (N_PIXELS,),
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
    "com_offset": (N_ACTUATORS,),
    "com_to_meas": (N_PIXELS, N_ACTUATORS),
}

MODAL_BASIS = modal_basis.Fourier()
# MODAL_BASIS = modal_basis.Zonal()
# MODAL_BASIS = modal_basis.Zernike()

# make sure that all named arrays have an entry in this dict:
for array_name in ARRAY_NAMES:
    assert array_name in ARRAY_SHAPES.keys()

INIT_VAL = {
    "meas_offset": 0.0,
    "meas_to_mode": 0.0,
    "filter_coeff_in": 0.0,
    "filter_coeff_out": 0.0,
    "mode_offset": 0.0,
    "mode_max": 1e6,
    "mode_min": -1e6,
    "mode_to_com": 0.0,
    "com_max": 1e6,
    "com_min": -1e6,
    "com_dist_buffer": 0.0,
    "com_offset": 0.0,
    "com_to_meas": 0.0,
}
# make sure that all named arrays have an entry in this dict:
for array_name in ARRAY_NAMES:
    assert array_name in INIT_VAL.keys()


class ZmqNoResponse(RuntimeError):
    """local error type for handling an offline RTC"""

    pass


@dataclass
class Beam:
    """Object for managing all interactions with the RTC at the per-beam level.
    """
    socket: Optional[ZmqReq] = field(init=False)
    beam_id: int
    host: str = DEFAULT_HOST

    def __post_init__(self):
        try:
            self.socket = self.get_zmq_socket()
        except:
            self.socket = None

    def get_zmq_socket(self) -> ZmqReq:
        if self.beam_id not in BEAM_TO_PORT:
            raise ValueError(
                f"Invalid beam {self.beam_id}. Expected one of {sorted(BEAM_TO_PORT)}"
            )
        port = BEAM_TO_PORT[self.beam_id]
        endpoint = f"tcp://{self.host}:{port}"
        print(f"Connecting to beam {self.beam_id} on {endpoint}")
        return ZmqReq(endpoint)

    def request(self, message: str):
        resp = self.socket.send_payload(message, is_str=True, decode_ascii=False)  # type: ignore
        if not isinstance(resp, dict):
            raise ZmqNoResponse(f"No valid response for command '{message}': {resp}")
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
        return path.join(BALDR_ROOT, "") # f"B{self.beam_id}_")

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
                f"meas_offset.shape incorrect\nexpected: {TARGET_SHAPE}, got: {array.shape}"
            )
        fits.writeto(filename=filename, data=array.astype(DTYPE), overwrite=True)

    def update_array(self, *, name: str, array: NDArray, push_rtc: bool = True):
        """Convenience function for updating RTC parameters from numpy arrays"""
        self.check_name(name)
        self.writefits(name=name, array=array)
        if push_rtc:
            self.request(name)

    def update_delay(self, delay: float):
        """Update the delay from a float"""
        self.request(f"delay {delay}")

    ############################################################
    ### High level functions for executing supervisory tasks ###
    ############################################################

    def reset(self, *, init: bool = False, push_rtc: bool = True):
        for name in ARRAY_NAMES:
            if init:
                self.writefits(name=name, array=None)
        for name in ARRAY_NAMES:
            if push_rtc:
                self.request(name)

    def set_leaky_gain_leak(
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

    def set_polc_gain(
        self,
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
        self.update_array(name="filter_coeff_in", array=filter_coeff_in)
        self.update_array(name="filter_coeff_out", array=filter_coeff_out)

    def flatten_dm(self):
        # compute filter coeffs. Zeros correspond to an "all stop" filter.
        filter_coeff_in = np.zeros([FILTER_LEN, N_MODES])
        filter_coeff_out = np.zeros([FILTER_LEN, N_MODES])
        # save coeffs to fits files
        self.update_array(name="filter_coeff_in", array=filter_coeff_in)
        self.update_array(name="filter_coeff_out", array=filter_coeff_out)

    def set_com_clip(self, *, clip_val: float):
        # compute filter coeffs. Zeros correspond to an "all stop" filter.
        com_max = clip_val * np.ones([N_ACTUATORS])
        # save coeffs to fits files
        self.update_array(name="com_max", array=com_max)
        self.update_array(name="com_min", array=-com_max)

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
            mode[i] = poke
            self.poke(array=mode)
            meas_pos = self.avg_meas(navg=navg, after_frame=CNT_MIN)
            # poke mode i (negative poke)
            mode[i] = -poke
            self.poke(array=mode)
            meas_neg = self.avg_meas(navg=navg, after_frame=CNT_MIN)
            meas = (meas_pos - meas_neg) / (2 * poke) * MEAS_SCALE

            # inject it to matrix
            mode_to_meas[:, i] = meas
        self.flatten_offsets()
        fits.writeto("DEBUG_mode_to_meas.fits", mode_to_meas, overwrite=True)
        return (mode_to_meas, -ref_meas)

    @staticmethod
    def build_meas_to_mode(
        *,
        mode_to_meas: NDArray,
        alpha: float,
        nmodes: Optional[int],
    ) -> NDArray:
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
        self,
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
        mode_to_com = MODAL_BASIS.modes_on_unit_disk(
            nsamplex=N_ACTX, nmodes=N_MODES
        )

        ### Measure mode_to_slope interaction
        # flatten DM
        self.flatten_dm()
        self.flatten_offsets()

        # set mode_to_com
        self.update_array(name="mode_to_com", array=mode_to_com)

        # measure modal imat
        mode_to_meas, meas_offset = self.measure_interaction_matrix(
            navg=navg,
            poke=poke,
            nmodes=nmodes,
        )

        # run some statistics on the measured interaction matrix
        # TODO

        # produce com imat
        if nmodes is None:
            nmodes = N_MODES
        com_to_meas = (
            mode_to_meas[:, :nmodes]
            @ np.linalg.solve(
                mode_to_com[:, :nmodes].T @ mode_to_com[:, :nmodes] + beta * np.eye(nmodes),
                mode_to_com[:, :nmodes].T,
            )
            / MEAS_SCALE
        )
        self.update_array(name="com_to_meas", array=com_to_meas)

        ### Invert mode_to_slope to build slope_to_mode reconstructor
        meas_to_mode = self.build_meas_to_mode(
            mode_to_meas=mode_to_meas, alpha=alpha, nmodes=nmodes
        )
        self.update_array(name="meas_to_mode", array=meas_to_mode)
        self.update_array(name="meas_offset", array=meas_offset)

    def create_leaky_matrices(
        self,
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
        mode_to_com = MODAL_BASIS.modes_on_unit_disk(
            nsamplex=N_ACTX, nmodes=N_MODES
        )

        ### Measure mode_to_slope interaction
        # flatten DM
        self.flatten_dm()
        self.flatten_offsets()

        # set mode_to_com
        self.update_array(name="mode_to_com", array=mode_to_com)

        # measure modal imat
        mode_to_meas, meas_offset = self.measure_interaction_matrix(
            navg=navg,
            poke=poke,
            nmodes=nmodes,
        )

        self.update_array(name="com_to_meas", array=np.zeros((N_PIXELS, N_ACTUATORS)))

        ### Invert mode_to_slope to build slope_to_mode reconstructor
        meas_to_mode = self.build_meas_to_mode(
            mode_to_meas=mode_to_meas, alpha=alpha, nmodes=nmodes
        )
        self.update_array(name="meas_to_mode", array=meas_to_mode)
        self.update_array(name="meas_offset", array=meas_offset)


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
        "--poke", help="value to poke each mode, try 0.01", type=float, default=POKE
    )

    parser.add_argument("--nmodes", help="maximum mode index to control", type=int)

    args = parser.parse_args()

    action_performed = False

    beam = Beam(beam_id=args.beam)

    if args.init is not None:
        print("initing!")
        try:
            beam.reset(init=True)
        except ZmqNoResponse:
            print("""
Succesfullly initialised arrays and wrote them to disk, but didn't
update them on the live RTC.

This is correct behaviour if the RTC is not yet running.
""")
        action_performed = True

    if args.clipcom is not None:
        beam.set_com_clip(clip_val=args.clipcom)
        action_performed = True

    if args.disturb is not None:
        if args.disturboff is not None:
            raise ValueError("cannot simultaneously be disturbing and not disturbing")
        # The default disturbance is a sine wave that sweeps accross the dm
        # over 20 frames.
        _, xx = np.meshgrid(
            np.linspace(0, 2 * 2 * np.pi, N_ACTX + 1)[:-1],
            np.linspace(0, 2 * 2 * np.pi, N_ACTX + 1)[:-1],
            indexing="ij",
        )
        xx_flat = xx.flatten()
        disturbance = np.zeros([N_ACTUATORS, DIST_LEN])
        for i, t in enumerate(np.linspace(0, 2 * np.pi, DIST_LEN + 1)[:-1]):
            disturbance[:, i] = 0.1 * np.sin(xx_flat + t)
        beam.update_array(name="com_dist_buffer", array=disturbance)
        action_performed = True

    if args.disturboff is not None:
        disturbance = np.zeros([N_ACTUATORS, DIST_LEN])
        beam.update_array(name="com_dist_buffer", array=disturbance)
        action_performed = True

    if args.polc is not None:
        if args.recompute is not None:
            beam.create_polc_matrices(nmodes=args.nmodes, poke=args.poke)
        if args.gain is not None:
            gain = args.gain
        else:
            gain = 0.3
        if args.leak is not None:
            leak = args.leak
        else:
            leak = 1.0
        beam.set_polc_gain(ewma_gain=gain, ewma_leak=leak)
        action_performed = True
    elif args.leaky is not None:
        if args.recompute is not None:
            beam.create_leaky_matrices(nmodes=args.nmodes, poke=args.poke)
        if args.gain is not None:
            gain = args.gain
        else:
            gain = 0.3
        if args.leak is not None:
            leak = args.leak
        else:
            leak = 0.95
        beam.set_leaky_gain_leak(gain=gain, leak=leak)
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
