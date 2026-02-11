from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, List, Optional, Sequence, Tuple, Callable, Protocol
import numpy as np
from baldr_python_rtc.baldr_rtc.io.base import CameraIO, DMIO


_EPS = 1e-12 # a global small number to avoid 1/0 errors

class ServoState(IntEnum):
    SERVO_OPEN = 0
    SERVO_CLOSE = 1


class MainState(IntEnum):
    SERVO_RUN = 0
    SERVO_STOP = 2





# ----------------------------
# Small helper: robust matmul
# ----------------------------
def _matmul(A: Any, x: Any) -> np.ndarray:
    """Best-effort A @ x using numpy, tolerating Python lists."""
    A_ = np.asarray(A)
    x_ = np.asarray(x).reshape(-1)
    return (A_ @ x_).reshape(-1)


# ----------------------------
# Core config sub-structs
# ----------------------------
@dataclass
class Limits:
    # legacy fields from C++ RTC
    open_on_flux_limit: float = 0.0
    close_on_strehl_limit: float = 0.0
    open_on_strehl_limit: float = 0.0
    open_on_dm_limit: float = 0.0
    LO_offload_limit: float = 0.0


@dataclass
class InjSignal:
    enabled: bool = False


@dataclass
class StateConfig:
    #  legacy fields from C++ RTC
    # C++ ctrl_model state-ish fields
    DM_flat: str = ""
    signal_space: str = ""          # "pix" | "dm" (legacy expects string)
    LO: int = 0
    controller_type: str = "unknown"
    inverse_method_LO: str = ""
    inverse_method_HO: str = ""
    phasemask: str = "H1"

    auto_close: int = 0
    auto_open: int = 1
    auto_tune: int = 0

    take_telemetry: bool = False    # kept for your python runtime
    auto_close_python: bool = False # optional if you want it later

    simulation_mode: int = 0        # legacy C++ sets 0 in read-in


@dataclass
class Matrices:
    # again legacy fields from C++ RTC
    # Matrices can be nested lists from TOML; leave them as Any
    I2A: Any = field(default_factory=list)
    I2M: Any = field(default_factory=list)
    I2M_LO: Any = field(default_factory=list)
    I2M_HO: Any = field(default_factory=list)
    M2C: Any = field(default_factory=list)
    M2C_LO: Any = field(default_factory=list)
    M2C_HO: Any = field(default_factory=list)
    I2rms_sec: Any = field(default_factory=list)
    I2rms_ext: Any = field(default_factory=list)

    # sizes (legacy)
    szm: int = 0
    sza: int = 0
    szp: int = 0


@dataclass
class ReferencePupils:
    # legacy fields from C++ RTC
    # flattened vectors (lists from TOML are fine)
    I0: Any = field(default_factory=list)
    N0: Any = field(default_factory=list)
    norm_pupil: Any = field(default_factory=list)

    # DM-space projections
    norm_pupil_dm: Any = field(default_factory=list)
    I0_dm: Any = field(default_factory=list)

    intrn_flx_I0: float = 1.0

    def project_to_dm(self, I2A: Any) -> None:
        # project both norm_pupil and I0
        self.norm_pupil_dm = _matmul(I2A, self.norm_pupil)
        self.I0_dm = _matmul(I2A, self.I0)

    def project_I0_to_dm(self, I2A: Any) -> None:
        self.I0_dm = _matmul(I2A, self.I0)

    def project_N0norm_to_dm(self, I2A: Any) -> None:
        self.norm_pupil_dm = _matmul(I2A, self.norm_pupil)


@dataclass
class Pixels:
    # legacy fields from C++ RTC
    # crop_pixels is [r1, r2, c1, c2]
    crop_pixels: Any = field(default_factory=list)

    bad_pixels: Any = field(default_factory=list)
    pupil_pixels: Any = field(default_factory=list)
    interior_pixels: Any = field(default_factory=list)
    secondary_pixels: Any = field(default_factory=list)
    exterior_pixels: Any = field(default_factory=list)

    def validate(self) -> None:
        if self.crop_pixels is None:
            raise RuntimeError("Pixels.crop_pixels is None")
        try:
            n = len(self.crop_pixels)
        except Exception:
            raise RuntimeError("Pixels.crop_pixels is not a sized sequence")
        if n != 4:
            raise RuntimeError(f"Pixels.crop_pixels must have 4 elements, got {n}")


@dataclass
class Filters:
    # again legacy fields from C++ RTC
    # These are often 0/1 masks (but keep flexible)
    bad_pixel_mask: Any = field(default_factory=list)
    pupil: Any = field(default_factory=list)
    secondary: Any = field(default_factory=list)
    exterior: Any = field(default_factory=list)
    inner_pupil_filt: Any = field(default_factory=list)
    # NEW strehl filt (not legacy)
    strehl_filt: Any = field(default_factory=list)

    opd_m_interc: float = 0 
    opd_m_slope_1: float = 0 
    opd_m_slope_2: float = 0 
    opd_m_x_knee: float = 0  

    # DM-space projections
    bad_pixel_mask_dm: Any = field(default_factory=list)
    pupil_dm: Any = field(default_factory=list)
    secondary_dm: Any = field(default_factory=list)
    exterior_dm: Any = field(default_factory=list)
    inner_pupil_filt_dm: Any = field(default_factory=list)

    def project_to_dm(self, I2A: Any) -> None:
        self.bad_pixel_mask_dm = _matmul(I2A, self.bad_pixel_mask)
        self.pupil_dm = _matmul(I2A, self.pupil)
        self.secondary_dm = _matmul(I2A, self.secondary)
        self.exterior_dm = _matmul(I2A, self.exterior)
        self.inner_pupil_filt_dm = _matmul(I2A, self.inner_pupil_filt)


@dataclass
class CamConfig:
    # Keep these as strings to match your legacy intent
    fps: str = ""
    gain: str = ""
    testpattern: str = ""

    bias: str = ""
    flat: str = ""
    imagetags: str = ""
    led: str = ""
    events: str = ""
    extsynchro: str = ""
    rawimages: str = ""
    cooling: str = ""
    mode: str = ""
    resetwidth: str = ""
    nbreadworeset: str = ""
    cropping: str = ""
    cropping_columns: str = ""
    cropping_rows: str = ""
    aduoffset: str = ""


# ----------------------------
# Master config object
# ----------------------------
@dataclass
class BDRConfig:
    # high-level camera/rtc settings (python-side)
    fps: float = 0.0

    # legacy-ish sub-structs
    matrices: Matrices = field(default_factory=Matrices)
    reference_pupils: ReferencePupils = field(default_factory=ReferencePupils)
    pixels: Pixels = field(default_factory=Pixels)
    filters: Filters = field(default_factory=Filters)
    cam: CamConfig = field(default_factory=CamConfig)

    limits: Limits = field(default_factory=Limits)
    inj_signal: InjSignal = field(default_factory=InjSignal)
    state: StateConfig = field(default_factory=StateConfig)

    # --- IO (your python additions; keep these) ---
    io_mode: str = "null"  # null | shm | zmq | simulation
    io_beam: int = 1

    # SHM (xaosim.shmlib)
    io_cam_path: str = "/dev/shm/baldr{beam}.im.shm"
    io_dm_path: str = "/dev/shm/dm{beam}.im.shm"
    io_shm_nosem: bool = True
    io_shm_semid: int = 0

    # ZMQ
    io_zmq_cam_addr: str = "tcp://127.0.0.1:5556"
    io_zmq_dm_addr: str = "tcp://127.0.0.1:5555"
    io_zmq_timeout_ms: int = 200

    # Null backend
    io_null_shape: Tuple[int, int] = (32, 32)

    def validate(self) -> None:
        print('not implemented')
        #self.pixels.validate()
        # You can add more checks later (matrix shape checks, etc.)





class Controller(Protocol):
    def process(self, e: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class RTCModel:
    # config-ish
    signal_space: str  # "pix" | "dm"

    # matrices (already numpy arrays, float64/float32; whatever you choose)
    I2A: Optional[np.ndarray]      # (n_dm_sig, n_pix_sig) or None if pix space
    I2M_LO: np.ndarray             # (n_lo, n_sig)
    I2M_HO: np.ndarray             # (n_ho, n_sig)
    M2C_LO: np.ndarray             # (n_act, n_lo)
    M2C_HO: np.ndarray             # (n_act, n_ho)

    # runtime references (already in *signal space*)
    N0_runtime: np.ndarray         # (n_sig,)
    i_setpoint_runtime: np.ndarray # (n_sig,)  # e.g. I_ref = I0/N0
    inner_pupil_filt: np.ndarray # (n_sig,) - used to filter N0 pupil for agregating in i_setpoint_runtime normalization (see build_rtc_model() in baldr_rtc_python/baldr_rtc/server.py)
    strehl_filt: np.ndarray # (n_sig,) 
    dark: np.ndarray # n_pix
    # controllers (must expose .process(vec)->vec)
    ctrl_LO: Controller
    ctrl_HO: Controller
    # optional hooks (keep None for “fast path”)
    process_frame: Optional[Callable[[np.ndarray], np.ndarray]] = None
    perf_model: Optional[Callable[[np.ndarray], np.ndarray]] = None
    perf_param: Optional[object] = None


@dataclass
class RuntimeGlobals:
    beam: int = 0
    phasemask: str = "H1"
    
    active_config_filename: str | None = None
    mode: str = "unknown" # This is required for status which wag needs 
    servo_mode: MainState = MainState.SERVO_RUN
    servo_mode_LO: ServoState = ServoState.SERVO_OPEN
    servo_mode_HO: ServoState = ServoState.SERVO_OPEN
    pause_rtc: bool = False

    # IO handles (wired by server at startup)
    camera_io: Optional[CameraIO] = None
    dm_io: Optional[DMIO] = None

    # this holds the RTC config state. 
    rtc_config: BDRConfig = field(default_factory=BDRConfig)

    # this will hold the runtime RTC matricies and variables
    model: Optional[RTCModel] = None # we dont want a default creation yet

    
    



### previous version 
# @dataclass
# class Matrices:
#     I2M_LO: list = field(default_factory=list)
#     I2M_HO: list = field(default_factory=list)


# @dataclass
# class Limits:
#     open_on_flux_limit: float = 0.0
#     close_on_strehl_limit: float = 0.0
#     open_on_strehl_limit: float = 0.0


# @dataclass
# class InjSignal:
#     enabled: bool = False


# @dataclass
# class StateConfig:
#     controller_type: str = "unknown"
#     auto_close: bool = False
#     take_telemetry: bool = False


# @dataclass
# class BDRConfig:
#     fps: float = 0.0
#     matrices: Matrices = field(default_factory=Matrices)
#     limits: Limits = field(default_factory=Limits)
#     inj_signal: InjSignal = field(default_factory=InjSignal)
#     state: StateConfig = field(default_factory=StateConfig)

#     # --- IO ---
#     # Where camera frames and DM commands come from/go to.
#     io_mode: str = "null"  # null | shm | zmq
#     io_beam: int = 1

#     # SHM (xaosim.shmlib)
#     io_cam_path: str = "/dev/shm/baldr{beam}.im.shm"
#     io_dm_path: str = "/dev/shm/dm{beam}.im.shm"
#     io_shm_nosem: bool = True
#     io_shm_semid: int = 0

#     # expected shapes (rows, cols)
#     io_null_shape: Tuple[int, int] = (48, 48)
#     io_shm_shape: Tuple[int, int] = (48, 48)
#     io_zmq_shape: Tuple[int, int] = (48, 48)

#     # ZMQ
#     io_zmq_cam_addr: str = "tcp://127.0.0.1:5556"
#     io_zmq_dm_addr: str = "tcp://127.0.0.1:5555"
#     io_zmq_timeout_ms: int = 100
    
    

#     def init_derived_parameters(self) -> None:
#         if self.fps <= 0:
#             self.fps = 1000.0


# @dataclass
# class RuntimeGlobals:
#     beam: int = 0
#     phasemask: str = "H4"
    
#     active_config_filename: str | None = None
#     mode: str = "unknown" # This is required for status which wag needs 
#     servo_mode: MainState = MainState.SERVO_RUN
#     servo_mode_LO: ServoState = ServoState.SERVO_OPEN
#     servo_mode_HO: ServoState = ServoState.SERVO_OPEN
#     pause_rtc: bool = False

#     # IO handles (wired by server at startup)
#     camera_io: Optional[CameraIO] = None
#     dm_io: Optional[DMIO] = None

#     # this holds the current RTC state. 
#     rtc_config: BDRConfig = field(default_factory=BDRConfig)


    
    

