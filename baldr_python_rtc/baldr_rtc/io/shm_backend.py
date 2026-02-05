# baldr_rtc/io/shm_backend.py
from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Sequence, Union
import numpy as np
import glob
import time 
import os
import subprocess
from .cam_client import CamClient

from .base import Frame, CameraIO, DMIO

try:
    from xaosim.shmlib import shm
except Exception:  # pragma: no cover
    shm = None



def _git_root() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def _default_dm_shapes_dir() -> Path:
    # matches your old convention
    return _git_root() / "DMShapes"



@dataclass(frozen=True)
class ShmPaths:
    cam_path: str
    dm_path: str


def default_shm_paths(beam: int) -> ShmPaths:
    # Real system convention (Linux): ImageStreamIO-style files in /dev/shm
    return ShmPaths(
        cam_path=f"/dev/shm/baldr{beam}.im.shm",
        dm_path=f"/dev/shm/dm{beam}.im.shm",
    )

class ShmCameraIO(CameraIO):
    """
    Camera frames from ImageStreamIO/xaosim SHM.

    Notes:
    - On Linux in production, you typically want nosem=False and a known semid.
    - On Mac for development, /dev/shm usually doesn't exist -> cam.empty=True.
    """

    def __init__(self, 
                 path: str, 
                 *, 
                 nosem: bool = True, 
                 semid: Optional[int] = None,
                 cam_client: Optional[CamClient] = None,
                 cam_cmd_pad_to: Optional[int] = None):
        if shm is None:
            raise ImportError("xaosim.shmlib.shm could not be imported (xaosim not installed?)")

        self.path = path
        self.nosem = bool(nosem)
        self.semid = semid
        self._shm = shm(path, nosem=self.nosem)
        self._cam_client = cam_client
        self._cam_cmd_pad_to = cam_cmd_pad_to

    @property
    def empty(self) -> bool:
        return bool(getattr(self._shm, "empty", False))


    def catch_up_with_sem(self): 
        
        if self.semid is not None:
            self._shm.catch_up_with_sem(int(self.semid))
        else:
            print( "self.semid = None, cannot catchup to anything ")

    # This is the CameraIO base function (USE THIS IN RTC! )
    def get_frame(self, *, timeout_s: Optional[float] = None) -> Frame:
        # NOTE: xaosim/shmlib doesn't expose a timeout in this API.
        # If semid is configured, the SHM call is already "blocking for next frame" behavior.
        data = np.asarray(self._shm.get_latest_data_slice(semid=self.semid))
        frame_id = int(self._shm.get_counter())
        return Frame(data=data, t_s=time.time(), frame_id=frame_id)
    

    def get_latest_data_slice(self) -> np.ndarray:
        """
        Returns latest frame as a numpy array (2D for typical camera).
        If semid is set and semaphores are enabled, blocks until next post.
        """
        
        return np.asarray(self._shm.get_latest_data_slice(semid=self.semid))
    


    def get_data(self) -> np.ndarray:
        """
        Returns the cred 1 bufffer (typically 200 frames) as a numpy array (2D for typical camera).
        If semid is set and semaphores are enabled, blocks until next post.
        """
        return self._shm.get_latest_data(semid=self.semid)


    # ---- ZMQ control-plane helpers (NOT for RTC per-frame) ----
    def send_cam_cmd(self, cmd: str) -> str:
        if self._cam_client is None:
            print("\n===!!!===\nsend_cam_cmd failed!\nNo CamClient attached to ShmCameraIO\n===!!!===\n")
            return None
        else:
            return self._cam_client.send_command(cmd, pad_to=self._cam_cmd_pad_to)


    
    def get_camera_config(self) -> Dict:
        if self._cam_client is None:
            print("\n===!!!===\nget_camera_config failed!\nNo CamClient attached to ShmCameraIO\n===!!!===\n")
            return None
        else:
            return self._cam_client.get_camera_config()
    

    def close(self) -> None:
        # keep erase_file False for safety
        self._shm.close(erase_file=False)


############## DM 
@dataclass(frozen=True)
class DMShmPaths:
    combined: str
    sub_channels: List[str]


def default_dm_shm_paths(beam: int) -> DMShmPaths:
    beam = int(beam)
    sub = sorted(glob.glob(f"/dev/shm/dm{beam}disp*.im.shm"))
    combined = f"/dev/shm/dm{beam}.im.shm"
    return DMShmPaths(combined=combined, sub_channels=sub)


class ShmDMIO(DMIO):
    """
    Wrapper around Frantz/Asgard DM SHM layout.

    Layout:
      - sub-channels: /dev/shm/dm{beam}disp*.im.shm
      - combined:     /dev/shm/dm{beam}.im.shm  (used to post sems after updates)

    Conventions (kept from dmclass):
      - channel 0: factory flat / calibrated flat
      - channel 1: calibration shapes (e.g. cross)
      - channel main_chn (default=2): user modes / general commands
    """

    def __init__(
        self,
        beam: int,
        *,
        main_chn: int = 2,
        nosem: bool = False,
        shapes_dir: Optional[Union[str, Path]] = None,
    ):
        if shm is None:
            raise ImportError("xaosim.shmlib.shm not available (xaosim not installed?)")

        beam = int(beam)
        if beam not in (1, 2, 3, 4):
            raise ValueError("beam must be 1..4")

        self.beam = beam
        self.main_chn = int(main_chn)
        self.nosem = bool(nosem)

        self.paths = default_dm_shm_paths(beam)
        self.shapes_dir = Path(shapes_dir) if shapes_dir is not None else _default_dm_shapes_dir()

        # sub-channel SHMs
        self.shms: List[Any] = []
        for p in self.paths.sub_channels:
            self.shms.append(shm(p, nosem=self.nosem))

        self.nch = len(self.shms)

        # combined SHM (for post_sems)
        if self.nch == 0:
            raise RuntimeError(
                f"No DM SHM sub-channels found for beam={beam}. "
                f"Expected something like /dev/shm/dm{beam}disp*.im.shm. "
                f"Is the DM server running?"
            )

        self.shm0 = shm(self.paths.combined, nosem=self.nosem)

        if not (0 <= self.main_chn < self.nch):
            raise ValueError(f"main_chn={self.main_chn} out of range (nch={self.nch})")

    # ---------- file selection (same mapping as old dmclass) ----------

    def select_flat_cmd(self) -> Path:
        flat_cmd_files = {
            1: "17DW019#113_FLAT_MAP_COMMANDS.txt",
            2: "17DW019#053_FLAT_MAP_COMMANDS.txt",
            3: "17DW019#093_FLAT_MAP_COMMANDS.txt",
            4: "17DW019#122_FLAT_MAP_COMMANDS.txt",
        }
        return self.shapes_dir / flat_cmd_files[self.beam]

    def select_flat_cmd_offset(self) -> Path:
        # most recent BEAM{beam}_FLAT_MAP*.txt in shapes_dir
        pattern = str(self.shapes_dir / f"BEAM{self.beam}_FLAT_MAP*.txt")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No flat offset file matching {pattern}")
        latest = max(matches, key=os.path.getmtime)
        return Path(latest)

    # ---------- mapping / formatting ----------

    @staticmethod
    def cmd_2_map2D(cmd140: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """
        Convert a 140-vector into 12x12 with missing corners inserted.

        Inserts at [0, 10, 130, 140] to create length 144, then reshape to (12,12).
        """
        cmd = np.asarray(cmd140).reshape(-1)
        if cmd.size != 140:
            raise ValueError(f"Expected 140 elements, got {cmd.size}")
        return np.insert(cmd, [0, 10, 130, 140], fill).reshape((12, 12))

    @staticmethod
    def _as_12x12(cmd: np.ndarray) -> np.ndarray:
        """
        Accepts:
          - 140 vector -> converted to 12x12 (fill=0)
          - 144 vector -> reshaped to 12x12
          - 12x12 -> passed through
        """
        arr = np.asarray(cmd)
        if arr.shape == (12, 12):
            return arr
        if arr.size == 140:
            return ShmDMIO.cmd_2_map2D(arr, fill=0.0)
        if arr.size == 144:
            return arr.reshape((12, 12))
        raise ValueError(f"Unsupported DM cmd shape {arr.shape} / size {arr.size}")

    def _post(self) -> None:
        # trigger combined shm sem(s) so DM server updates
        ### IS THIS THE RIGHT SEM (1)? DOUBLE cHECK 
        self.shm0.post_sems(1)

    # ---------- high-level actions (kept compatible) ----------

    def activate_flat(self) -> None:
        flat = np.loadtxt(self.select_flat_cmd())
        self.shms[0].set_data(self.cmd_2_map2D(flat, fill=0.0))
        self._post()

    def activate_calibrated_flat(self) -> None:
        flat = np.loadtxt(self.select_flat_cmd())
        off = np.loadtxt(self.select_flat_cmd_offset())
        self.shms[0].set_data(self.cmd_2_map2D(flat + off, fill=0.0))
        self._post()

    def get_baldr_flat_offset(self) -> np.ndarray:
        return np.loadtxt(self.select_flat_cmd_offset())

    def activate_cross(self, amp: float = 0.1) -> None:
        dms = 12
        ii0 = dms // 2 - 1
        cross = np.zeros((dms, dms), dtype=float)
        cross[ii0 : ii0 + 2, :] = amp
        cross[:, ii0 : ii0 + 2] = amp
        self.shms[1].set_data(cross)
        self._post()

    def apply_modes(self, amplitude_list: Sequence[float], basis_list: Sequence[np.ndarray]) -> None:
        if len(amplitude_list) != len(basis_list):
            raise ValueError("amplitude_list and basis_list must have the same length")
        cmd = np.zeros((12, 12), dtype=float)
        for a, m in zip(amplitude_list, basis_list):
            cmd += float(a) * self._as_12x12(m)
        self.shms[self.main_chn].set_data(cmd)
        self._post()

    def set_data(self, cmd: np.ndarray) -> None:
        """
        Writes to main_chn by default (dmclass behavior).
        Accepts 140, 144, or 12x12.
        """
        self.shms[self.main_chn].set_data(self._as_12x12(cmd))
        self._post()

    ### USE THis METHOD (base.py class structure)
    def write(self, cmd: np.ndarray) -> None:
        self.set_data(cmd)


    def set_channel(self, chn: int, cmd: np.ndarray) -> None:
        """
        Optional: write explicitly to a given sub-channel.
        """
        chn = int(chn)
        if not (0 <= chn < self.nch):
            raise ValueError(f"chn={chn} out of range (nch={self.nch})")
        self.shms[chn].set_data(self._as_12x12(cmd))
        self._post()

    def zero_all(self) -> None:
        zeros = np.zeros((12, 12), dtype=float)
        for ss in self.shms:
            ss.set_data(zeros)
        self._post()

    def close(self) -> None:
        for ss in self.shms:
            ss.close(erase_file=False)
        self.shms.clear()
        if getattr(self, "shm0", None) is not None:
            self.shm0.close(erase_file=False)
            self.shm0 = None