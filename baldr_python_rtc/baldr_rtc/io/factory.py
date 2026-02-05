from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import CameraIO, DMIO
from .null_backend import NullCameraIO, NullDMIO
from .cam_client import CamClient

@dataclass(frozen=True)
class IOHandles:
    camera: CameraIO
    dm: DMIO


def make_io(cfg, beam: int) -> IOHandles:
    """Create IO backends from *flat* fields on BDRConfig.

    The config parser populates:
      cfg.io_mode: "null" | "shm" | "zmq"
      cfg.io_cam_path, cfg.io_dm_path, cfg.io_shm_nosem, cfg.io_shm_semid, cfg.io_shm_shape
      cfg.io_zmq_cam_addr, cfg.io_zmq_dm_addr, cfg.io_zmq_timeout_ms, cfg.io_zmq_shape
      cfg.io_null_shape
    """

    mode = str(getattr(cfg, "io_mode", "null")).strip().lower()

    if mode == "shm":
        from .shm_backend import ShmCameraIO, ShmDMIO
        from .cam_client import CamClient

        cam_path_tmpl = str(getattr(cfg, "io_cam_path", "/dev/shm/baldr{beam}.im.shm"))
        cam_path = cam_path_tmpl.format(beam=int(beam))

        nosem = bool(getattr(cfg, "io_shm_nosem", True))
        semid_raw = getattr(cfg, "io_shm_semid", None)
        semid: Optional[int] = None if nosem else (int(semid_raw) if semid_raw is not None else 0)

        # Optional control-plane client (safe to omit in pure RTC)
        use_cam_client = bool(getattr(cfg, "io_cam_use_client", False))
        cam_client = CamClient() if use_cam_client else None

        cam = ShmCameraIO(
            path=cam_path,
            nosem=nosem,
            semid=semid,
            cam_client=cam_client,
            cam_cmd_pad_to=getattr(cfg, "io_cam_cmd_pad_to", None),
        )

        dm = ShmDMIO(
            beam=int(beam),
            main_chn=int(getattr(cfg, "io_dm_main_chn", 2)),
            nosem=bool(getattr(cfg, "io_dm_nosem", False)),
            shapes_dir=getattr(cfg, "io_dm_shapes_dir", None),
        )

        return IOHandles(camera=cam, dm=dm)
    
    elif mode == "zmq":
        from .zmq_backend import ZMQCameraIO, ZMQDMIO

        cam = ZMQCameraIO(
            addr=str(getattr(cfg, "io_zmq_cam_addr", "tcp://127.0.0.1:5556")),
            timeout_ms=int(getattr(cfg, "io_zmq_timeout_ms", 200)),
            shape=tuple(getattr(cfg, "io_zmq_shape", (48, 48))),
        )
        dm = ZMQDMIO(
            addr=str(getattr(cfg, "io_zmq_dm_addr", "tcp://127.0.0.1:5555")),
            timeout_ms=int(getattr(cfg, "io_zmq_timeout_ms", 200)),
        )
        return IOHandles(camera=cam, dm=dm)

    elif mode == "simulation":

        """
        This is a simulator mode that doesnt use shared memory backend so it can 
        be run simply on other OS beyond Linux.

        Its fragile on set up for specific tests. Not a general mode to rely on! 

        will require 
        # baldrapp 
        python -m pip install BaldrApp
        # pyzelda fork (double check dependancy in https://pypi.org/project/BaldrApp/)
        python -m pip install pyzelda@git+https://github.com/courtney-barrer/pyZELDA.git@b42aaea5c8a47026783a15391df5e058360ea15e 
        # aotools 
        python -m pip install aotools
        # maybe also pandas, scikit-image if not included above 
        python -m pip install pandas, scikit-image


        """
        from .simulation_backend import (
            BaldrAppSimConfig,
            BaldrAppSimState,
            SimCameraIO,
            SimDMIO,
        )

        sim_cfg = BaldrAppSimConfig(
            use_pyZelda=bool(getattr(cfg, "io_sim_use_pyzelda", False)),
            fps=float(getattr(cfg, "io_sim_fps", 1730.0)),
            binning=int(getattr(cfg, "io_sim_binning", 6)),
            ron=float(getattr(cfg, "io_sim_ron", 12.0)),
            qe=float(getattr(cfg, "io_sim_qe", 0.7)),
            # add more knobs later only if/when you need them
        )

        state = BaldrAppSimState(cfg=sim_cfg)
        cam = SimCameraIO(state)
        dm = SimDMIO(state)

        return IOHandles(camera=cam, dm=dm)
        

    # default: null
    shape = tuple(getattr(cfg, "io_null_shape", (32, 32)))
    return IOHandles(camera=NullCameraIO(shape=shape), dm=NullDMIO())



# from __future__ import annotations

# from .null_backend import NullCamera, NullDM, NullIOConfig


# def make_io(cfg, *, zwfs_ns=None, zwfs_ns_tmp=None, detector=None):
#     # cfg can be SimpleNamespace or dict-like; keep this forgiving.
#     io = getattr(cfg, "io", None) or {}
#     mode = (getattr(io, "mode", None) or io.get("mode", "null")).lower()

#     if mode == "null":
#         null_cfg = NullIOConfig(
#             shape=tuple(getattr(getattr(io, "null", {}), "shape", (96, 96)) if hasattr(io, "null") else io.get("null", {}).get("shape", (96, 96))),
#             noise_std=float(getattr(getattr(io, "null", {}), "noise_std", 0.0) if hasattr(io, "null") else io.get("null", {}).get("noise_std", 0.0)),
#         )
#         return NullCamera(null_cfg), NullDM()

#     if mode == "baldrapp":
#         if zwfs_ns is None:
#             raise ValueError("io.mode='baldrapp' requires zwfs_ns to be provided by your server setup.")
#         from .baldrapp_backend import BaldrAppSimBackend, BaldrAppSimConfig
#         bcfg_section = getattr(io, "baldrapp", None) or (io.get("baldrapp", {}) if isinstance(io, dict) else {})
#         use_pyZelda = bool(getattr(bcfg_section, "use_pyZelda", False) if hasattr(bcfg_section, "use_pyZelda") else bcfg_section.get("use_pyZelda", False))
#         backend = BaldrAppSimBackend(zwfs_ns, zwfs_ns_tmp=zwfs_ns_tmp, detector=detector, cfg=BaldrAppSimConfig(use_pyZelda=use_pyZelda))
#         return backend, backend  # (cam, dm)

#     raise ValueError(f"Unknown io.mode={mode!r}")