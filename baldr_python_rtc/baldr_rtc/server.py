from __future__ import annotations

import queue
import threading
import numpy as np
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from baldr_python_rtc.baldr_rtc.core.config import readBDRConfig
from baldr_python_rtc.baldr_rtc.core.state import RuntimeGlobals, RTCModel
from baldr_python_rtc.baldr_rtc.io.factory import make_io
from baldr_python_rtc.baldr_rtc.telemetry.ring import TelemetryRingBuffer
from baldr_python_rtc.baldr_rtc.telemetry.worker import TelemetryThread, TelemetryWriter
from baldr_python_rtc.baldr_rtc.rtc.loop import RTCThread
from baldr_python_rtc.baldr_rtc.commander.server import CommanderServer
from baldr_python_rtc.baldr_rtc.commander.commands import build_commander_module
from baldr_python_rtc.baldr_rtc.rtc.controllers import build_controller

def _print_banner(*, beam: int, socket: str, cfg_path: str, cfg) -> None:
    # Keep this pure printing: no side effects beyond stdout.
    banner = r"""
  ____    _    _      ____   ____      ____  _____  ____ 
 | __ )  / \  | |    |  _ \ |  _ \    |  _ \|_   _|/ ___|
 |  _ \ / _ \ | |    | | | || |_) |   | |_) | | | | |    
 | |_) / ___ \| |___ | |_| ||  _ <    |  _ <  | | | |___ 
 |____/_/   \_\_____||____/ |_| \_\   |_| \_\ |_|  \____|

                 BALDR RTC (Python)
"""
    print(banner)



def inspect_rtc_model(model: Any, *, name: str = "g.model") -> None:
    """
    For each field in `model`:
      a) print type
      b) try print size/shape/len
      c) if numeric array-like, print quantiles [0,25,50,75,100] of flattened finite values
         else if string/list-of-strings, print it.
    """
    def _is_str_seq(x: Any) -> bool:
        return isinstance(x, (list, tuple)) and all(isinstance(v, str) for v in x)

    def _maybe_shape(x: Any) -> Optional[str]:
        try:
            if hasattr(x, "shape"):
                return str(tuple(getattr(x, "shape")))
        except Exception:
            pass
        return None

    def _maybe_len(x: Any) -> Optional[int]:
        try:
            return len(x)  # type: ignore[arg-type]
        except Exception:
            return None

    def _try_numeric_quantiles(x: Any) -> Optional[np.ndarray]:
        # Return quantiles for numeric array-like, else None.
        # Avoid copying unless needed; still safe for typical sizes.
        try:
            arr = np.asarray(x)
        except Exception:
            return None

        if arr.dtype == object:
            return None

        if not np.issubdtype(arr.dtype, np.number):
            return None

        flat = arr.ravel()
        if flat.size == 0:
            return np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=float)

        # Use finite subset to avoid NaN/inf poisoning.
        try:
            finite = flat[np.isfinite(flat)]
        except Exception:
            finite = flat

        if finite.size == 0:
            return np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=float)

        # np.quantile is fine here; if you want *ultra* fast, swap to percentiles on float32.
        q = np.quantile(finite.astype(np.float64, copy=False), [0.0, 0.25, 0.50, 0.75, 1.0])
        return q

    def _items(obj: Any) -> Iterable[tuple[str, Any]]:
        if obj is None:
            return []
        if is_dataclass(obj):
            # Use vars() for speed; includes only dataclass fields for dataclasses with slots too.
            try:
                return list(vars(obj).items())
            except TypeError:
                # slots without __dict__
                d = asdict(obj)
                return list(d.items())
        if isinstance(obj, Mapping):
            return list(obj.items())
        # fallback: introspect public attributes
        names = [n for n in dir(obj) if not n.startswith("_")]
        out = []
        for n in names:
            try:
                out.append((n, getattr(obj, n)))
            except Exception:
                out.append((n, "<unreadable>"))
        return out

    if model is None:
        print(f"{name}: None")
        return

    print(f"=== inspect {name} ===")
    for k, v in _items(model):
        tname = type(v).__name__
        shape = _maybe_shape(v)
        ln = _maybe_len(v)

        size_str = ""
        if shape is not None:
            size_str = f"shape={shape}"
        elif ln is not None and not isinstance(v, (str, bytes)):
            size_str = f"len={ln}"

        print(f"\n[{k}] type={tname}" + (f" {size_str}" if size_str else ""))

        # Strings / list-of-strings
        if isinstance(v, str):
            print(f"  value={v!r}")
            continue
        if _is_str_seq(v):
            print(f"  value={list(v)!r}")
            continue

        # Numeric quantiles
        q = _try_numeric_quantiles(v)
        if q is not None:
            print(f"  q0,q25,q50,q75,q100 = {q}")
            continue

        # Fallback: brief repr
        try:
            r = repr(v)
        except Exception:
            r = "<unreprable>"
        if len(r) > 400:
            r = r[:400] + "…"
        print(f"  repr={r}")



def build_rtc_model(cfg) -> RTCModel:
    st = cfg.state
    space = (st.signal_space or "pix").strip().lower()

    I2A = np.asarray(cfg.matrices.I2A, dtype=float) # if space == "dm" else None

    I2M_LO = np.asarray(cfg.matrices.I2M_LO, dtype=float)
    I2M_HO = np.asarray(cfg.matrices.I2M_HO, dtype=float)
    M2C_LO = np.asarray(cfg.matrices.M2C_LO, dtype=float)
    M2C_HO = np.asarray(cfg.matrices.M2C_HO, dtype=float)

    # filters 
    inner_pupil_filt = np.asarray( cfg.filters.inner_pupil_filt, dtype=bool).reshape(-1) 

    # reduction 
    #dark = np.asarray(cfg.matrices.M2C_HO, dtype=float)
    # references from legacy toml
    I0 = np.asarray(cfg.reference_pupils.I0, dtype=float).reshape(-1)
    N0 = np.asarray(cfg.reference_pupils.N0, dtype=float).reshape(-1)
    dark = np.asarray(cfg.reference_pupils.dark, dtype=float).reshape(-1)
    if space == "dm":
        # NOTE: assumes I0/N0 are already in the SAME reduced pixel vector space as I2A expects
        I0 = I2A @ I0
        N0 = I2A @ N0
        inner_pupil_filt = (I2A @ inner_pupil_filt).astype(bool)
    
    i_setpoint_runtime = I0 / np.mean( N0[inner_pupil_filt]  ) 
    N0_runtime = N0
    



    # controllers
    n_LO = int(I2M_LO.shape[0])
    n_HO = int(I2M_HO.shape[0])
    dt = 1/1000 # WE SHOULD NOT USE dt IN CONTROLLERS! 

    ct = cfg.state.controller_type.strip().lower()
    if ct == "pid":
        ctrl_LO = build_controller("pid", n_LO, dt=dt, kp=0, ki=0, kd=0, u_min=None, u_max=None)
        ctrl_HO = build_controller("pid", n_HO, dt=dt, kp=0, ki=0, kd=0, u_min=None, u_max=None)
    elif ct == "leaky":
        ctrl_LO = build_controller("leaky", n_LO, rho=1.0, ki=0, kp=0, u_min=None, u_max=None)
        ctrl_HO = build_controller("leaky", n_HO, rho=1.0, ki=0, kp=0, u_min=None, u_max=None)
    else:
        print(f"controller_type {ct} is not implemented\n!!!!!!!!!! JUT CONTINUE WITH PID , FIX THIS LATER")
        ctrl_LO = build_controller("pid", n_LO, dt=dt, kp=0, ki=0, kd=0, u_min=None, u_max=None)
        ctrl_HO = build_controller("pid", n_HO, dt=dt, kp=0, ki=0, kd=0, u_min=None, u_max=None)
    
    return RTCModel(
        signal_space=space,
        I2A=I2A,
        I2M_LO=I2M_LO,
        I2M_HO=I2M_HO,
        M2C_LO=M2C_LO,
        M2C_HO=M2C_HO,
        N0_runtime=N0_runtime,
        dark=dark.reshape(32,32),
        inner_pupil_filt=inner_pupil_filt,
        i_setpoint_runtime=i_setpoint_runtime,
        ctrl_LO=ctrl_LO,   # or g.ctrl_LO if you store controllers in globals
        ctrl_HO=ctrl_HO,
        process_frame=None,
        perf_model=None,
        perf_param=None,
    )



def _print_runtime_info(*, g: RuntimeGlobals, socket: str) -> None:
    cfg = g.rtc_config

    # IO handle types are often the most truthful + useful
    cam_name = type(g.camera_io).__name__ if g.camera_io is not None else "None"
    dm_name  = type(g.dm_io).__name__ if g.dm_io is not None else "None"

    print("Runtime:")
    print(f"  beam:           {g.beam}")
    print(f"  phasemask:      {g.phasemask}")
    print(f"  commander:      {socket}")
    print(f"  config:         {g.active_config_filename}")
    print(f"  io_mode:        {getattr(cfg, 'io_mode', 'unknown')}")
    print(f"  camera_io:      {cam_name}")
    print(f"  dm_io:          {dm_name}")
    print(f"  fps:            {cfg.fps}")
    print(f"  controller:     {cfg.state.controller_type}")
    print(f"  telemetry:      {'ON' if cfg.state.take_telemetry else 'OFF'}")
    print(f"  servo:          {g.servo_mode.name}")
    print(f"  LO:             {g.servo_mode_LO.name}")
    print(f"  HO:             {g.servo_mode_HO.name}")
    print("")


def main(
    *,
    beam: int,
    phasemask: str,
    socket: str,
    config_path: str,
    telem_dir: str,
    telem_capacity: int,
    flush_hz: float,
    chunk_seconds: float,
    debug: bool,
) -> int:
    stop_event = threading.Event()
    cmd_queue: "queue.Queue[dict]" = queue.Queue()

    cfg = readBDRConfig(config_path=config_path, beam=beam, phasemask=phasemask)
    
    _print_banner(beam=beam, socket=socket, cfg_path=str(config_path), cfg=cfg)

    g = RuntimeGlobals(beam=beam, 
                       phasemask=phasemask,
                       active_config_filename=config_path, 
                       rtc_config=cfg)

    print(f"\n---\n...setting up camera and DM object in {cfg.io_mode} mode")
    io = make_io(cfg, beam=beam)
    g.camera_io = io.camera
    g.dm_io = io.dm
    print("finished setting up camera and DM object\n---\n")

    g.model = build_rtc_model(cfg) # all the rtc runtime goodies that gets used in loop

    _print_runtime_info(g=g, socket=socket)
    if debug:
        inspect_rtc_model(g.model)

    #ring = TelemetryRingBuffer(capacity=telem_capacity)

    # dimensions (all ints)
    n_sig = g.model.i_setpoint_runtime.size
    n_lo  = g.model.I2M_LO.shape[0]
    n_ho  = g.model.I2M_HO.shape[0]
    n_act = g.model.M2C_LO.shape[0]

    # raw pixel count depends on signal space
    if g.model.signal_space.lower() == "pix":
        n_pix = n_sig
    else:
        # dm space: raw pixels still come from camera
        # assume camera frame is flattened before push
        # safest: infer from camera backend
        n_pix = 32*32 #g.camera_io.shape[0] * g.camera_io.shape[1]

    ring = TelemetryRingBuffer(
        capacity=telem_capacity,
        n_pix=n_pix,
        n_sig=n_sig,
        n_lo=n_lo,
        n_ho=n_ho,
        n_act=n_act,
        store_ctrl_state=True,
    )
    
    # make telem also in globals 'g' so we can poll via commander (for gui use)
    g.telem_ring = ring
    
    writer = TelemetryWriter(out_dir=telem_dir, beam=beam)
    telem_thread = TelemetryThread(
        globals_=g,
        ring=g.telem_ring,
        writer=writer,
        stop_event=stop_event,
        flush_hz=flush_hz,
        chunk_seconds=chunk_seconds,
    )

    rtc_thread = RTCThread(globals_=g, command_queue=cmd_queue, telem_ring=g.telem_ring, stop_event=stop_event)

    module = build_commander_module(globals_=g, command_queue=cmd_queue, stop_event=stop_event)
    cmd_server = CommanderServer(endpoint=socket, module=module, stop_event=stop_event)

    rtc_thread.start()
    telem_thread.start()
    cmd_server.start()

    stop_event.wait()

    # Best-effort close IO backends
    try:
        g.camera_io.close()
    except Exception:
        pass
    try:
        g.dm_io.close()
    except Exception:
        pass

    rtc_thread.join(timeout=2.0)
    telem_thread.join(timeout=2.0)
    cmd_server.join(timeout=2.0)
    return 0

