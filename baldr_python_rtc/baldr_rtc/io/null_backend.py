
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .base import Frame, CameraIO, DMIO


@dataclass
class NullCameraIO(CameraIO):
    shape: Tuple[int, int] = (32, 32)
    dtype: np.dtype = np.float32
    _frame_id: int = 0

    @property
    def input_shape(self) -> Tuple[int, int]:
        # Some parts of RTC expect camera_io.input_shape
        return self.shape

    def get_frame(self, *, timeout_s: Optional[float] = None) -> Frame:
        self._frame_id += 1
        data = np.zeros(self.shape, dtype=self.dtype)
        return Frame(data=data, t_s=time.time(), frame_id=self._frame_id)

    def close(self) -> None:
        return None


@dataclass
class NullDMIO(DMIO):
    n_act: int = 140
    last_cmd: Optional[np.ndarray] = None

    def write(self, cmd: np.ndarray) -> None:
        self.last_cmd = np.asarray(cmd, dtype=float).reshape(-1)

    def close(self) -> None:
        return None
# first attempt 
# from __future__ import annotations

# import time
# from dataclasses import dataclass
# from typing import Optional, Tuple

# import numpy as np

# from .base import Frame, CameraIO, DMIO


# @dataclass
# class NullCameraIO(CameraIO):
#     """In-process dummy camera.

#     Useful for running the RTC without any external IO.
#     """

#     shape: Tuple[int, int] = (32, 32)
#     dtype: np.dtype = np.float32
#     _frame_id: int = 0

#     def get_frame(self, *, timeout_s: Optional[float] = None) -> Frame:
#         self._frame_id += 1
#         data = np.zeros(self.shape, dtype=self.dtype)
#         return Frame(data=data, t_s=time.time(), frame_id=self._frame_id)

#     def close(self) -> None:
#         return None


# @dataclass
# class NullDMIO(DMIO):
#     """In-process dummy DM sink."""

#     n_act: int = 140
#     last_cmd: Optional[np.ndarray] = None

#     def write(self, cmd: np.ndarray) -> None:
#         self.last_cmd = np.asarray(cmd, dtype=float).reshape(-1)

#     def close(self) -> None:
#         return None

