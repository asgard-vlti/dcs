from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import zmq

from .base import Frame, CameraIO, DMIO

# prob redundant this one
@dataclass
class ZMQCameraIO(CameraIO):
    """REQ/REP camera backend.

    Assumes the server returns a flat bytes payload for the frame. For now this
    backend is intentionally simple: it lets you bring up a simulator quickly
    and evolve the message format later.
    """

    socket: str
    shape: tuple[int, int]
    dtype: str = "float32"
    timeout_ms: int = 1000

    def __post_init__(self) -> None:
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REQ)
        self._sock.connect(self.socket)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._frame_id = 0

    def get_frame(self) -> Frame:
        self._sock.send_string("GET_FRAME")
        payload = self._sock.recv()
        arr = np.frombuffer(payload, dtype=np.dtype(self.dtype)).reshape(self.shape)
        self._frame_id += 1
        return Frame(data=arr, t_s=time.time(), frame_id=self._frame_id)

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass


@dataclass
class ZMQDMIO(DMIO):
    socket: str
    timeout_ms: int = 1000

    def __post_init__(self) -> None:
        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.REQ)
        self._sock.connect(self.socket)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

    def write(self, cmd: np.ndarray) -> None:
        cmd = np.asarray(cmd)
        self._sock.send_multipart([b"SET_DM", cmd.astype("float32").tobytes()])
        _ = self._sock.recv()  # ACK

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
