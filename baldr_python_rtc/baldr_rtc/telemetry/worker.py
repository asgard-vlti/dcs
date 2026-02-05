from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from baldr_python_rtc.baldr_rtc.core.state import MainState, RuntimeGlobals
from baldr_python_rtc.baldr_rtc.telemetry.ring import TelemetryRingBuffer, TelemetryChunk


@dataclass
class TelemetryWriter:
    out_dir: str
    beam: int

    def __post_init__(self) -> None:
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)


    def write_chunk(self, chunk: TelemetryChunk) -> Path:
        ts = time.time()
        p = Path(self.out_dir) / f"beam{self.beam}_telem_{ts:.3f}.npz"

        payload = {
            # --- per-sample metadata ---
            "frame_id": chunk.frame_id,
            "t_s": chunk.t_s,
            "lo_state": chunk.lo_state,
            "ho_state": chunk.ho_state,
            "paused": chunk.paused,

            # --- signals ---
            "i_raw": chunk.i_raw,
            "i_space": chunk.i_space,
            "s": chunk.s,

            # --- errors / controls ---
            "e_lo": chunk.e_lo,
            "e_ho": chunk.e_ho,
            "u_lo": chunk.u_lo,
            "u_ho": chunk.u_ho,

            # --- DM space ---
            "c_lo": chunk.c_lo,
            "c_ho": chunk.c_ho,
            "cmd": chunk.cmd,

            # --- metrics ---
            "opd_metric": chunk.opd_metric,
            "snr_metric": chunk.snr_metric,

            # --- misc ---
            "overruns": np.array([chunk.overruns], dtype=np.int64),
        }

        # Optional: controller internal state snapshots (if present)
        if chunk.ctrl_state_lo is not None:
            payload["ctrl_state_lo"] = chunk.ctrl_state_lo
        if chunk.ctrl_state_ho is not None:
            payload["ctrl_state_ho"] = chunk.ctrl_state_ho

        np.savez_compressed(p, **payload)
        return p

    # def write_chunk(self, chunk: TelemetryChunk) -> Path:
    #     ts = time.time()
    #     p = Path(self.out_dir) / f"beam{self.beam}_telem_{ts:.3f}.npz"
    #     np.savez_compressed(
    #         p,
    #         frame_id=chunk.frame_id,
    #         t_s=chunk.t_s,
    #         lo_state=chunk.lo_state,
    #         ho_state=chunk.ho_state,
    #         paused=chunk.paused,
    #         metric_flux=chunk.metric_flux,
    #         metric_strehl=chunk.metric_strehl,
    #         overruns=np.array([chunk.overruns], dtype=np.int64),
    #     )
    #     return p

    def write_static(self, *, g: RuntimeGlobals) -> Path:
        ts = time.time()
        p = Path(self.out_dir) / f"beam{self.beam}_static_{ts:.3f}.npz"
        m = g.model
        if m is None:
            # No model yet; still write minimal identity so tooling works
            np.savez_compressed(
                p,
                beam=np.array([g.beam], dtype=np.int32),
                phasemask=np.array([g.phasemask], dtype=object),
                config=np.array([getattr(g, "active_config_filename", "")], dtype=object),
            )
            return p

        np.savez_compressed(
            p,
            beam=np.array([g.beam], dtype=np.int32),
            phasemask=np.array([g.phasemask], dtype=object),
            config=np.array([getattr(g, "active_config_filename", "")], dtype=object),
            signal_space=np.array([m.signal_space], dtype=object),

            I2A=m.I2A if m.I2A is not None else np.empty((0, 0), dtype=np.float32),
            I2M_LO=m.I2M_LO,
            I2M_HO=m.I2M_HO,
            M2C_LO=m.M2C_LO,
            M2C_HO=m.M2C_HO,

            N0_runtime=m.N0_runtime,
            i_setpoint_runtime=m.i_setpoint_runtime,
        )
        return p


class TelemetryThread(threading.Thread):
    def __init__(
        self,
        globals_: RuntimeGlobals,
        ring: TelemetryRingBuffer,
        writer: TelemetryWriter,
        stop_event: threading.Event,
        *,
        flush_hz: float = 1.0,
        chunk_seconds: float = 2.0,
    ):
        super().__init__(daemon=True)
        self.g = globals_
        self.ring = ring
        self.writer = writer
        self.stop_event = stop_event
        self.flush_dt = 1.0 / max(flush_hz, 1e-6)
        self.chunk_seconds = max(chunk_seconds, 0.1)



    def run(self) -> None:
        last_static_key = None

        while not self.stop_event.is_set():
            if self.g.servo_mode == MainState.SERVO_STOP:
                self.stop_event.set()
                break

            if not self.g.rtc_config.state.take_telemetry:
                time.sleep(self.flush_dt)
                continue

            # --- write static once per "model identity" ---
            # Prefer an explicit model_version if you add one later.
            mv = getattr(self.g, "model_version", None)
            model_id = int(mv) if mv is not None else (id(self.g.model) if self.g.model is not None else 0)

            static_key = (
                getattr(self.g, "active_config_filename", ""),
                str(getattr(self.g, "phasemask", "")),
                model_id,
            )

            if static_key != last_static_key:
                self.writer.write_static(g=self.g)
                last_static_key = static_key

            # --- chunking ---
            fps = float(self.g.rtc_config.fps) if self.g.rtc_config.fps > 0 else 1000.0
            max_samples = int(self.chunk_seconds * fps)

            chunk = self.ring.pop_chunk(max_samples=max_samples)
            if chunk is not None:
                self.writer.write_chunk(chunk)

            time.sleep(self.flush_dt)
            
