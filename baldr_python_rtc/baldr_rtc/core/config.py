from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from baldr_python_rtc.baldr_rtc.core.state import BDRConfig, Limits, InjSignal, StateConfig


def _load_toml(config_path: str) -> Dict[str, Any]:
    p = Path(config_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    txt = p.read_text()
    try:
        import tomllib  # py>=3.11
        return tomllib.loads(txt)
    except Exception:
        import toml
        return toml.loads(txt)


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# top level to decide what read-in mode we use (legacy or otherwise)
def readBDRConfig(config_path: str, *, beam: int = 1, phasemask: str = "H1") -> BDRConfig:
    data = _load_toml(config_path)

    beam_key = f"beam{int(beam)}"
    #Legacy format if beam table exists
    if beam_key in data and isinstance(data[beam_key], dict):
        return readBDRConfig_legacy(config_path, beam=beam, phasemask=phasemask)

    # otherwise: your existing simple reader
    return readBDRConfig_simple(config_path, beam=beam, phasemask=phasemask)



def readBDRConfig_simple(config_path: str, *, beam: int, phasemask: str) -> BDRConfig:
    data = _load_toml(config_path)
    cfg = BDRConfig()

    # keep your existing simple parsing...
    cfg.state.phasemask = phasemask

    # Example: simple IO
    cfg.io_mode = str(_get(data, "io", "mode", default=cfg.io_mode))
    cfg.io_beam = int(_get(data, "io", "beam", default=beam))
    cfg.io_cam_path = str(_get(data, "io", "cam_path", default=cfg.io_cam_path))
    cfg.io_dm_path = str(_get(data, "io", "dm_path", default=cfg.io_dm_path))
    cfg.io_shm_nosem = bool(_get(data, "io", "shm_nosem", default=cfg.io_shm_nosem))
    cfg.io_shm_semid = int(_get(data, "io", "shm_semid", default=cfg.io_shm_semid))

    cfg.io_null_shape = tuple(
        _get(data, "io", "null_shape", default=list(cfg.io_null_shape))
    )

    #cfg.init_derived_parameters()
    return cfg



# Full legacy read in from Toml structure read into C++ during baldr comissioning 1 (https://github.com/mikeireland/dcs/blob/main/baldr/baldr.cpp)
def readBDRConfig_legacy(config_path: str, *, beam: int, phasemask: str) -> BDRConfig:
    data = _load_toml(config_path)
    cfg = BDRConfig()
    cfg.state.phasemask = phasemask

    beam_key = f"beam{int(beam)}"
    beam_tbl = data.get(beam_key)
    if not isinstance(beam_tbl, dict):
        raise RuntimeError(f"Beam configuration not found for key: {beam_key}")

    phase_tbl = beam_tbl.get(phasemask)
    if not isinstance(phase_tbl, dict):
        raise RuntimeError(f'Phase mask "{phasemask}" not found under {beam_key}')

    ctrl_tbl = phase_tbl.get("ctrl_model")
    if not isinstance(ctrl_tbl, dict):
        raise RuntimeError(f'Missing ctrl_model table under {beam_key}.{phasemask}')

    # ---- state ----
    cfg.state.DM_flat = str(ctrl_tbl.get("DM_flat", ""))
    cfg.state.signal_space = str(ctrl_tbl.get("signal_space", ""))
    cfg.state.LO = int(ctrl_tbl.get("LO", 0) or 0)
    cfg.state.controller_type = str(ctrl_tbl.get("controller_type", ""))
    cfg.state.inverse_method_LO = str(ctrl_tbl.get("inverse_method_LO", ""))
    cfg.state.inverse_method_HO = str(ctrl_tbl.get("inverse_method_HO", ""))
    cfg.state.auto_close = int(ctrl_tbl.get("auto_close", 0) or 0)
    cfg.state.auto_open = int(ctrl_tbl.get("auto_open", 1) or 1)

    # replicate buggy-legacy behavior, but tolerate either spelling
    auto_tuen = ctrl_tbl.get("auto_tuen", None)
    auto_tune = ctrl_tbl.get("auto_tune", None)
    cfg.state.auto_tune = int(auto_tuen if auto_tuen is not None else (auto_tune if auto_tune is not None else 0))

    cfg.state.simulation_mode = 0  # matches C++

    # ---- pixels ----
    # these are typically 1D arrays of ints
    cfg.pixels.crop_pixels = ctrl_tbl["crop_pixels"]
    cfg.pixels.pupil_pixels = ctrl_tbl["pupil_pixels"]
    cfg.pixels.bad_pixels = ctrl_tbl["bad_pixels"]
    cfg.pixels.interior_pixels = ctrl_tbl["interior_pixels"]
    cfg.pixels.secondary_pixels = ctrl_tbl["secondary_pixels"]
    cfg.pixels.exterior_pixels = ctrl_tbl["exterior_pixels"]
    cfg.pixels.validate()



    # ---- filters ----
    cfg.filters.bad_pixel_mask = ctrl_tbl["bad_pixel_mask"]
    cfg.filters.pupil = ctrl_tbl["pupil"]
    cfg.filters.secondary = ctrl_tbl["secondary"]
    cfg.filters.exterior = ctrl_tbl["exterior"]
    cfg.filters.inner_pupil_filt = ctrl_tbl["inner_pupil_filt"]
    # NEW (from asgard-alignment/calibration/build_strehl_model_v2.py)
    #import numpy as np
    #cfg.filters.strehl_filt = np.zeros(32*32)
    cfg.filters.strehl_filt = ctrl_tbl["strehl_filter"]

    # ---- matrices ----
    cfg.matrices.szm = int(ctrl_tbl.get("szm", 0) or 0)
    cfg.matrices.sza = int(ctrl_tbl.get("sza", 0) or 0)
    cfg.matrices.szp = int(ctrl_tbl.get("szp", 0) or 0)

    cfg.matrices.I2A = ctrl_tbl["I2A"]
    cfg.matrices.I2M_LO = ctrl_tbl["I2M_LO"]
    cfg.matrices.I2M_HO = ctrl_tbl["I2M_HO"]
    cfg.matrices.M2C = ctrl_tbl["M2C"]
    cfg.matrices.M2C_LO = ctrl_tbl["M2C_LO"]
    cfg.matrices.M2C_HO = ctrl_tbl["M2C_HO"]
    cfg.matrices.I2rms_sec = ctrl_tbl["I2rms_sec"]
    cfg.matrices.I2rms_ext = ctrl_tbl["I2rms_ext"]

    # ---- reference pupils ----
    cfg.reference_pupils.I0 = ctrl_tbl["I0"]
    cfg.reference_pupils.N0 = ctrl_tbl["N0"]
    cfg.reference_pupils.norm_pupil = ctrl_tbl["norm_pupil"]
    cfg.reference_pupils.intrn_flx_I0 = float(ctrl_tbl.get("intrn_flx_I0", 1.0) or 1.0)

    # ---- reduction ----
    cfg.reference_pupils.dark = ctrl_tbl["dark"]

    # projections (skip reduction, per your instruction)
    cfg.reference_pupils.project_to_dm(cfg.matrices.I2A)
    cfg.filters.project_to_dm(cfg.matrices.I2A)

    # ---- limits ----
    cfg.limits.close_on_strehl_limit = float(ctrl_tbl.get("close_on_strehl_limit", 0.0) or 0.0)
    cfg.limits.open_on_strehl_limit = float(ctrl_tbl.get("open_on_strehl_limit", 0.0) or 0.0)
    cfg.limits.open_on_flux_limit = float(ctrl_tbl.get("open_on_flux_limit", 0.0) or 0.0)
    cfg.limits.open_on_dm_limit = float(ctrl_tbl.get("open_on_dm_limit", 0.0) or 0.0)
    cfg.limits.LO_offload_limit = float(ctrl_tbl.get("LO_offload_limit", 0.0) or 0.0)

    # ---- camera_config ----
    cam_tbl = ctrl_tbl.get("camera_config", None)
    if isinstance(cam_tbl, dict):
        cfg.cam.fps = str(cam_tbl.get("fps", ""))
        cfg.cam.gain = str(cam_tbl.get("gain", ""))
        cfg.cam.testpattern = str(cam_tbl.get("testpattern", ""))
        cfg.cam.bias = str(cam_tbl.get("bias", ""))
        cfg.cam.flat = str(cam_tbl.get("flat", ""))
        cfg.cam.imagetags = str(cam_tbl.get("imagetags", ""))
        cfg.cam.led = str(cam_tbl.get("led", ""))
        cfg.cam.events = str(cam_tbl.get("events", ""))
        cfg.cam.extsynchro = str(cam_tbl.get("extsynchro", ""))
        cfg.cam.rawimages = str(cam_tbl.get("rawimages", ""))
        cfg.cam.cooling = str(cam_tbl.get("cooling", ""))
        cfg.cam.mode = str(cam_tbl.get("mode", ""))
        cfg.cam.resetwidth = str(cam_tbl.get("resetwidth", ""))
        cfg.cam.nbreadworeset = str(cam_tbl.get("nbreadworeset", ""))
        cfg.cam.cropping = str(cam_tbl.get("cropping", ""))
        cfg.cam.cropping_columns = str(cam_tbl.get("cropping_columns", ""))
        cfg.cam.cropping_rows = str(cam_tbl.get("cropping_rows", ""))
        cfg.cam.aduoffset = str(cam_tbl.get("aduoffset", ""))

    # ---- Python IO section (optional) ----
    cfg.io_mode = str(_get(data, "io", "mode", default=cfg.io_mode))
    cfg.io_beam = int(_get(data, "io", "beam", default=cfg.io_beam))
    cfg.io_cam_path = str(_get(data, "io", "cam_path", default=cfg.io_cam_path))
    cfg.io_dm_path = str(_get(data, "io", "dm_path", default=cfg.io_dm_path))
    cfg.io_shm_nosem = bool(_get(data, "io", "shm_nosem", default=cfg.io_shm_nosem))
    
    cfg.io_zmq_cam_addr = str(_get(data, "io", "zmq_cam_addr", default=cfg.io_zmq_cam_addr))
    cfg.io_zmq_dm_addr = str(_get(data, "io", "zmq_dm_addr", default=cfg.io_zmq_dm_addr))
    #cfg.io_zmq_timeout_ms = int(_get(data, "io", "zmq_timeout_ms", default=cfg.io_zmq_timeout_ms))
    #cfg.io_null_shape = tuple(_get(data, "io", "null_shape", default=list(cfg.io_null_shape)))


    #cfg.init_derived_parameters()
    return cfg



# ### The first run running 
# ## A lite weight way to read things in 
# def readBDRConfig(config_path: str,    
#                     *,
#                     beam: int,
#                     phasemask: str) -> BDRConfig:
#     p = Path(config_path).expanduser()
#     if not p.exists():
#         raise FileNotFoundError(f"Config file not found: {p}")

#     data: Dict[str, Any]
#     try:
#         import tomllib  # py>=3.11
#         data = tomllib.loads(p.read_text())
#     except Exception:
#         import toml  # fallback
#         data = toml.loads(p.read_text())

#     cfg = BDRConfig()

#     cfg.state = StateConfig(
#         controller_type=str(_get(data, "state", "controller_type", default="unknown")),
#         auto_close=bool(_get(data, "state", "auto_close", default=False)),
#         take_telemetry=bool(_get(data, "state", "take_telemetry", default=False)),
#     )
#     cfg.inj_signal = InjSignal(enabled=bool(_get(data, "inj_signal", "enabled", default=False)))
#     cfg.limits = Limits(
#         open_on_flux_limit=float(_get(data, "limits", "open_on_flux_limit", default=0.0)),
#         close_on_strehl_limit=float(_get(data, "limits", "close_on_strehl_limit", default=0.0)),
#         open_on_strehl_limit=float(_get(data, "limits", "open_on_strehl_limit", default=0.0)),
#     )

#     cam_fps = _get(data, "cam", "fps", default="0")
#     try:
#         cfg.fps = float(cam_fps)
#     except Exception:
#         cfg.fps = 0.0

#     # --- IO wiring (optional) ---
#     cfg.io_mode = str(_get(data, "io", "mode", default=cfg.io_mode))
#     cfg.io_beam = int(_get(data, "io", "beam", default=cfg.io_beam))
#     cfg.io_cam_path = str(_get(data, "io", "cam_path", default=cfg.io_cam_path))
#     cfg.io_dm_path = str(_get(data, "io", "dm_path", default=cfg.io_dm_path))
#     cfg.io_shm_nosem = bool(_get(data, "io", "shm_nosem", default=cfg.io_shm_nosem))
#     cfg.io_zmq_cam_addr = str(_get(data, "io", "zmq_cam_addr", default=cfg.io_zmq_cam_addr))
#     cfg.io_zmq_dm_addr = str(_get(data, "io", "zmq_dm_addr", default=cfg.io_zmq_dm_addr))
#     cfg.io_zmq_timeout_ms = int(_get(data, "io", "zmq_timeout_ms", default=cfg.io_zmq_timeout_ms))
#     cfg.io_null_shape = tuple(_get(data, "io", "null_shape", default=list(cfg.io_null_shape)))

#     cfg.init_derived_parameters()
#     return cfg


