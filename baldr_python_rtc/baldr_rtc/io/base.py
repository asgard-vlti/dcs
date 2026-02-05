from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Protocol



@dataclass(frozen=True)
class Frame:
    """Single camera frame + basic metadata."""

    data: np.ndarray
    t_s: float
    frame_id: int


class CameraIO(Protocol):
    """Minimal camera interface for the RTC loop."""
    def get_frame(self, *, timeout_s: Optional[float] = None) -> Frame: ...
    def close(self) -> None: ...

class DMIO(Protocol):
    """Minimal camera interface for the RTC loop."""
    def write(self, cmd: np.ndarray) -> None: ...
    def close(self) -> None: ...
# class CameraIO(Protocol):
#     """Minimal camera interface for the RTC loop."""

#     def get_frame(self, *, timeout_s: Optional[float] = None) -> Frame:
#         ...

#     def close(self) -> None:
#         ...


# class DMIO(Protocol):
#     """Minimal DM interface for the RTC loop."""

#     def write(self, cmd: np.ndarray) -> None:
#         ...

#     def close(self) -> None:
#         ...

