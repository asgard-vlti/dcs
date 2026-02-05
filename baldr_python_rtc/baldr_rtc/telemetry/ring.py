# baldr_rtc/telemetry/ring.py
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Optional
import numpy as np


@dataclass(slots=True)
class TelemetryChunk:
    frame_id: np.ndarray
    t_s: np.ndarray
    lo_state: np.ndarray
    ho_state: np.ndarray
    paused: np.ndarray

    i_raw: np.ndarray     # (N, n_pix)
    i_space: np.ndarray   # (N, n_sig)  (can be same as i_raw if pix space)
    s: np.ndarray         # (N, n_sig)

    e_lo: np.ndarray      # (N, n_lo)
    e_ho: np.ndarray      # (N, n_ho)
    u_lo: np.ndarray      # (N, n_lo)
    u_ho: np.ndarray      # (N, n_ho)

    c_lo: np.ndarray      # (N, n_act)
    c_ho: np.ndarray      # (N, n_act)
    cmd: np.ndarray       # (N, n_act)

    opd_metric: np.ndarray  
    snr_metric: np.ndarray
    # optional controller internal states (keep if you want “exact replay”)
    ctrl_state_lo: Optional[np.ndarray] = None
    ctrl_state_ho: Optional[np.ndarray] = None

    overruns: int = 0


class TelemetryRingBuffer:
    """
    Fixed-length preallocated ring for per-frame debug telemetry.
    - Single writer (RTC thread).
    - Many readers (GUI, telemetry thread).
    """
    def __init__(
        self,
        *,
        capacity: int,
        n_pix: int,
        n_sig: int,
        n_lo: int,
        n_ho: int,
        n_act: int,
        store_ctrl_state: bool = True,
        dtype: np.dtype = np.float32,
    ):
        self._total_written = 0          # monotonic count of samples ever pushed
        self._flushed_total = 0          # monotonic count of samples flushed to disk

        self.capacity = int(capacity)
        self.n_pix = int(n_pix)
        self.n_sig = int(n_sig)
        self.n_lo = int(n_lo)
        self.n_ho = int(n_ho)
        self.n_act = int(n_act)
        self.dtype = dtype
        self.store_ctrl_state = bool(store_ctrl_state)

        self._lock = Lock()
        self._w = 0
        self._count = 0
        self._overruns = 0

        # scalars
        self.frame_id = np.zeros(self.capacity, dtype=np.int64)
        self.t_s = np.zeros(self.capacity, dtype=np.float64)
        self.lo_state = np.zeros(self.capacity, dtype=np.int8)
        self.ho_state = np.zeros(self.capacity, dtype=np.int8)
        self.paused = np.zeros(self.capacity, dtype=np.int8)

        self.opd_metric = np.zeros(self.capacity, dtype=np.float32)
        self.snr_metric = np.zeros(self.capacity, dtype=np.float32)
        # vectors
        self.i_raw = np.zeros((self.capacity, self.n_pix), dtype=self.dtype)
        self.i_space = np.zeros((self.capacity, self.n_sig), dtype=self.dtype)
        self.s = np.zeros((self.capacity, self.n_sig), dtype=self.dtype)

        self.e_lo = np.zeros((self.capacity, self.n_lo), dtype=self.dtype)
        self.e_ho = np.zeros((self.capacity, self.n_ho), dtype=self.dtype)
        self.u_lo = np.zeros((self.capacity, self.n_lo), dtype=self.dtype)
        self.u_ho = np.zeros((self.capacity, self.n_ho), dtype=self.dtype)

        self.c_lo = np.zeros((self.capacity, self.n_act), dtype=self.dtype)
        self.c_ho = np.zeros((self.capacity, self.n_act), dtype=self.dtype)
        self.cmd = np.zeros((self.capacity, self.n_act), dtype=self.dtype)


        if self.store_ctrl_state:
            # keep generic: “controller state” can be any vector; store LO and HO separately
            self.ctrl_state_lo = np.zeros((self.capacity, self.n_lo), dtype=self.dtype)
            self.ctrl_state_ho = np.zeros((self.capacity, self.n_ho), dtype=self.dtype)
        else:
            self.ctrl_state_lo = None
            self.ctrl_state_ho = None

    def push(
        self,
        *,
        frame_id: int,
        t_s: float,
        lo_state: int,
        ho_state: int,
        paused: bool,
        i_raw: np.ndarray,
        i_space: np.ndarray,
        s: np.ndarray,
        e_lo: np.ndarray,
        e_ho: np.ndarray,
        u_lo: np.ndarray,
        u_ho: np.ndarray,
        c_lo: np.ndarray,
        c_ho: np.ndarray,
        cmd: np.ndarray,
        opd_metric: float,
        snr_metric: float,
        ctrl_state_lo: Optional[np.ndarray] = None,
        ctrl_state_ho: Optional[np.ndarray] = None,
    ) -> None:
        with self._lock:
            idx = self._w

            self.frame_id[idx] = int(frame_id)
            self.t_s[idx] = float(t_s)
            self.lo_state[idx] = int(lo_state)
            self.ho_state[idx] = int(ho_state)

            self.opd_metric[idx] = opd_metric
            self.snr_metric[idx] = snr_metric
            
            self.paused[idx] = 1 if paused else 0

            np.copyto(self.i_raw[idx], np.asarray(i_raw, dtype=self.dtype).reshape(-1))
            np.copyto(self.i_space[idx], np.asarray(i_space, dtype=self.dtype).reshape(-1))
            np.copyto(self.s[idx], np.asarray(s, dtype=self.dtype).reshape(-1))

            np.copyto(self.e_lo[idx], np.asarray(e_lo, dtype=self.dtype).reshape(-1))
            np.copyto(self.e_ho[idx], np.asarray(e_ho, dtype=self.dtype).reshape(-1))
            np.copyto(self.u_lo[idx], np.asarray(u_lo, dtype=self.dtype).reshape(-1))
            np.copyto(self.u_ho[idx], np.asarray(u_ho, dtype=self.dtype).reshape(-1))

            np.copyto(self.c_lo[idx], np.asarray(c_lo, dtype=self.dtype).reshape(-1))
            np.copyto(self.c_ho[idx], np.asarray(c_ho, dtype=self.dtype).reshape(-1))
            np.copyto(self.cmd[idx], np.asarray(cmd, dtype=self.dtype).reshape(-1))


            if self.store_ctrl_state:
                if ctrl_state_lo is not None:
                    np.copyto(self.ctrl_state_lo[idx], np.asarray(ctrl_state_lo, dtype=self.dtype).reshape(-1))
                if ctrl_state_ho is not None:
                    np.copyto(self.ctrl_state_ho[idx], np.asarray(ctrl_state_ho, dtype=self.dtype).reshape(-1))

            # advance ring
            self._w = (self._w + 1) % self.capacity
            if self._count < self.capacity:
                self._count += 1
            else:
                self._overruns += 1

            self._total_written += 1

    def snapshot_latest(self, n: int = 1) -> TelemetryChunk:
        """GUI-friendly: returns a *copy* of the latest n samples (no views)."""
        n = int(max(1, min(n, self._count)))
        with self._lock:
            end = self._w
            start = (end - n) % self.capacity
            if start < end:
                sl = slice(start, end)
                idx = None
            else:
                # wrapped: concatenate
                idx = np.r_[np.arange(start, self.capacity), np.arange(0, end)]
                sl = None

            def take(arr):
                if idx is not None:
                    return arr[idx].copy()
                return arr[sl].copy()

            return TelemetryChunk(
                frame_id=take(self.frame_id),
                t_s=take(self.t_s),
                opd_metric=take(self.opd_metric),
                snr_metric=take(self.snr_metric),
                lo_state=take(self.lo_state),
                ho_state=take(self.ho_state),
                paused=take(self.paused),
                i_raw=take(self.i_raw),
                i_space=take(self.i_space),
                s=take(self.s),
                e_lo=take(self.e_lo),
                e_ho=take(self.e_ho),
                u_lo=take(self.u_lo),
                u_ho=take(self.u_ho),
                c_lo=take(self.c_lo),
                c_ho=take(self.c_ho),
                cmd=take(self.cmd),
                ctrl_state_lo=take(self.ctrl_state_lo) if self.ctrl_state_lo is not None else None,
                ctrl_state_ho=take(self.ctrl_state_ho) if self.ctrl_state_ho is not None else None,
                overruns=int(self._overruns),
            )

    # def pop_chunk(self, max_samples: int) -> Optional[TelemetryChunk]:
    #     """
    #     Telemetry-thread usage: take up to max_samples latest samples as a chunk.
    #     This is non-destructive for the ring (we’re not trying to “consume”).
    #     """
    #     return self.snapshot_latest(n=max_samples)

    def pop_chunk(self, max_samples: int) -> Optional[TelemetryChunk]:
        """
        Telemetry-thread usage: return up to max_samples *new* samples since last pop.
        This advances an internal flush cursor. GUI polling should use snapshot_latest().
        """
        max_samples = int(max_samples)
        if max_samples <= 0:
            return None

        with self._lock:
            # n_new = self._total_written - self._flushed_total
            # if n_new <= 0:
            #     return None

            # n = min(max_samples, n_new)

            n_new = self._total_written - self._flushed_total
            if n_new <= 0:
                return None

            # If we fell behind and the ring overwrote old samples, fast-forward flush cursor
            max_retain = self._count  # <= capacity
            if n_new > max_retain:
                self._flushed_total = self._total_written - max_retain
                n_new = max_retain
            n = min(max_samples, n_new)
            # We want the oldest unflushed sample ... newest included in this chunk.
            # Compute absolute indices in [flushed_total, flushed_total + n)
            start_total = self._flushed_total
            end_total = start_total + n

            # Map those totals to ring indices.
            # Current write position self._w points to next write index.
            # The most recently written sample is at (self._w - 1).
            # The sample with total index (self._total_written - 1) is at (self._w - 1).
            # So total k maps to ring index:
            # idx = (self._w - (self._total_written - k)) % capacity
            cap = self.capacity

            def take_range(arr):
                out = np.empty((n,) + arr.shape[1:], dtype=arr.dtype) if arr.ndim > 1 else np.empty(n, dtype=arr.dtype)
                for j in range(n):
                    k = start_total + j
                    ridx = (self._w - (self._total_written - k)) % cap
                    out[j] = arr[ridx]
                return out

            chunk = TelemetryChunk(
                frame_id=take_range(self.frame_id),
                t_s=take_range(self.t_s),
                lo_state=take_range(self.lo_state),
                ho_state=take_range(self.ho_state),
                paused=take_range(self.paused),

                i_raw=take_range(self.i_raw),
                i_space=take_range(self.i_space),
                s=take_range(self.s),

                e_lo=take_range(self.e_lo),
                e_ho=take_range(self.e_ho),
                u_lo=take_range(self.u_lo),
                u_ho=take_range(self.u_ho),

                c_lo=take_range(self.c_lo),
                c_ho=take_range(self.c_ho),
                cmd=take_range(self.cmd),

                opd_metric=take_range(self.opd_metric),
                snr_metric=take_range(self.snr_metric),

                ctrl_state_lo=take_range(self.ctrl_state_lo) if self.ctrl_state_lo is not None else None,
                ctrl_state_ho=take_range(self.ctrl_state_ho) if self.ctrl_state_ho is not None else None,

                overruns=int(self._overruns),
            )

            self._flushed_total = end_total
            return chunk


#############
## version 1
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Optional
# import numpy as np


# @dataclass
# class TelemetryChunk:
#     frame_id: np.ndarray
#     t_s: np.ndarray
#     lo_state: np.ndarray
#     ho_state: np.ndarray
#     paused: np.ndarray
#     metric_flux: np.ndarray
#     metric_strehl: np.ndarray
#     overruns: int


# class TelemetryRingBuffer:
#     def __init__(self, capacity: int):
#         self.capacity = int(capacity)

#         self.frame_id = np.zeros(self.capacity, dtype=np.int64)
#         self.t_s = np.zeros(self.capacity, dtype=np.float64)
#         self.lo_state = np.zeros(self.capacity, dtype=np.int16)
#         self.ho_state = np.zeros(self.capacity, dtype=np.int16)
#         self.paused = np.zeros(self.capacity, dtype=np.int8)
#         self.metric_flux = np.zeros(self.capacity, dtype=np.float32)
#         self.metric_strehl = np.zeros(self.capacity, dtype=np.float32)

#         self._write_idx = 0
#         self._flush_idx = 0
#         self._overruns = 0

#     def push(
#         self,
#         *,
#         frame_id: int,
#         t_s: float,
#         lo_state: int,
#         ho_state: int,
#         paused: bool,
#         metric_flux: float,
#         metric_strehl: float,
#     ) -> None:
#         if (self._write_idx - self._flush_idx) >= self.capacity:
#             self._flush_idx = self._write_idx - (self.capacity - 1)
#             self._overruns += 1

#         i = self._write_idx % self.capacity
#         self.frame_id[i] = frame_id
#         self.t_s[i] = t_s
#         self.lo_state[i] = lo_state
#         self.ho_state[i] = ho_state
#         self.paused[i] = 1 if paused else 0
#         self.metric_flux[i] = metric_flux
#         self.metric_strehl[i] = metric_strehl

#         self._write_idx += 1

#     def available(self) -> int:
#         return max(0, self._write_idx - self._flush_idx)

#     def pop_chunk(self, max_samples: int) -> Optional[TelemetryChunk]:
#         n_avail = self.available()
#         if n_avail <= 0:
#             return None

#         n = int(min(max_samples, n_avail))
#         start = self._flush_idx

#         def copy_range(arr: np.ndarray) -> np.ndarray:
#             out = np.empty(n, dtype=arr.dtype)
#             for j in range(n):
#                 out[j] = arr[(start + j) % self.capacity]
#             return out

#         chunk = TelemetryChunk(
#             frame_id=copy_range(self.frame_id),
#             t_s=copy_range(self.t_s),
#             lo_state=copy_range(self.lo_state),
#             ho_state=copy_range(self.ho_state),
#             paused=copy_range(self.paused),
#             metric_flux=copy_range(self.metric_flux),
#             metric_strehl=copy_range(self.metric_strehl),
#             overruns=self._overruns,
#         )
#         self._flush_idx += n
#         return chunk
