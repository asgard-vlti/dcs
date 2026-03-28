from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Protocol

"""

------ CONGRATULATIONS 
if you made it here you're probably wondering why i set up the shm backend like this and not just use the
standard shmlib get_data, set_data methods in the main code. The reason is that my mac cannot properly run the shared memory like 
linux, so i wanted an agnostic backend (with standard DMIO amd CameraIO classes) that could run shm on the real system (mimir) 
in either operational or simulation mode, but also have a simulation mode that is more compatible across different OS that i 
can run at home on a mac  

have fun.

"""

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

