# baldr_rtc/io/baldrapp_backend.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from .base import Frame, CameraIO, DMIO

# THIS WILL BE A FRAGILE BACKEND USED BY BEN CB FOR SPECIFIC TESTS
# DO NOT RELY ON IT GENERALLY (YOU CAN ASK BEN FOR HELP IF YOU WANT TO USE IT)

@dataclass(frozen=True)
class BaldrAppSimConfig:
    use_pyZelda: bool = False
    fps: float = 1730.0
    binning: int = 6
    ron: float = 12.0
    qe: float = 0.7

    # atmosphere / AO1 defaults
    Nmodes_removed: int = 7
    it_lag: int = 10
    include_scintillation: bool = True
    jumps_per_iter: int = 1
    propagation_distance: float = 10000.0

    # star
    throughput: float = 1.0
    waveband: str = "H"
    magnitude: float = 1.0

    # turbulence
    r0_500: float = 0.10
    L0: float = 25.0
    r0_scint: float = 0.164
    L0_scint: float = 10.0
    random_seed: int = 2


class BaldrAppSimState:
    """
    Owns zwfs_ns + phase/scint screens + AO1 lag buffer + current amp/opd.
    CameraIO calls step() each frame; DMIO writes update zwfs_ns.dm.current_cmd.
    """

    def __init__(self, cfg: Optional[BaldrAppSimConfig] = None):
        self.cfg = cfg or BaldrAppSimConfig()

        import sys
        from pathlib import Path
        module_dir = Path('/Users/bencb/Documents/ASGARD/BaldrApp/')  # e.g. Path.home() / "projects/my_pkg/src"
        sys.path.insert(0, str(module_dir))  

        # ---- heavy imports live here (not at module import time) ----
        from types import SimpleNamespace
        import aotools
        import pyzelda.ztools as ztools
        from baldrapp.common import baldr_core as bldr
        from baldrapp.common import utilities as util
        from baldrapp.common import phasescreens as ps

        self.aotools = aotools
        self.ztools = ztools
        self.bldr = bldr
        self.util = util
        self.ps = ps

        # ---- hard-coded zwfs_ns default ----
        grid_ns = SimpleNamespace(
            telescope="solarstein",
            D=1.8,
            N=72,
            dim=72 * 4,
        )

        optics_ns = SimpleNamespace(
            wvl0=1.65e-6,
            F_number=21.2,
            mask_diam=1.06,
            theta=1.57079,
            coldstop_diam=8.4,
            coldstop_offset=(0.0, 0.0),
        )

        dm_ns = SimpleNamespace(
            dm_model="BMC-multi-3.5",
            actuator_coupling_factor=0.75,
            dm_pitch=1,
            dm_aoi=0.0,
            opd_per_cmd=3e-6,
            flat_rmse=0.0,
        )

        self.zwfs_ns = bldr.init_zwfs(grid_ns, optics_ns, dm_ns)
        self.zwfs_ns.stellar.bandwidth = 300  # nm

        self.detector = bldr.detector(
            binning=self.cfg.binning,
            dit=1.0 / float(self.cfg.fps),
            ron=float(self.cfg.ron),
            qe=float(self.cfg.qe),
        )
        self.zwfs_ns.detector = self.detector

        # ---- turbulence / scint ----
        self.dx = self.zwfs_ns.grid.D / self.zwfs_ns.grid.N
        wvl0 = self.zwfs_ns.optics.wvl0

        r0 = float(self.cfg.r0_500) * (wvl0 / 0.5e-6) ** (6.0 / 5.0)

        self.scrn = ps.PhaseScreenKolmogorov(
            nx_size=self.zwfs_ns.grid.dim,
            pixel_scale=self.dx,
            r0=r0,
            L0=float(self.cfg.L0),
            random_seed=int(self.cfg.random_seed),
        )

        self.scint_scrn = aotools.turbulence.infinitephasescreen.PhaseScreenVonKarman(
            nx_size=self.zwfs_ns.grid.dim,
            pixel_scale=self.dx,
            r0=float(self.cfg.r0_scint),
            L0=float(self.cfg.L0_scint),
            random_seed=int(self.cfg.random_seed),
        )

        # ---- AO1 basis + lag buffer ----
        pm = self.zwfs_ns.grid.pupil_mask.astype(bool)
        self.pm = pm

        basis_cropped = ztools.zernike.zernike_basis(
            nterms=max(int(self.cfg.Nmodes_removed), 5),
            npix=self.zwfs_ns.grid.N,
        )
        basis_template = np.zeros_like(self.zwfs_ns.grid.pupil_mask, dtype=float)
        self.basis = np.array([util.insert_concentric(np.nan_to_num(b, 0.0), basis_template) for b in basis_cropped])

        self.reco_list = [0.0 * self.pm for _ in range(int(self.cfg.it_lag))]

        # ---- star amplitude baseline + internal OPD ----
        amp0 = (
            float(self.cfg.throughput)
            * (np.pi * (self.zwfs_ns.grid.D / 2.0) ** 2)
            / (np.pi * (self.zwfs_ns.grid.N / 2.0) ** 2)
            * util.magnitude_to_photon_flux(
                magnitude=float(self.cfg.magnitude),
                band=str(self.cfg.waveband),
                wavelength=1e9 * wvl0,
            )
        ) ** 0.5

        self.amp_input_0 = amp0 * self.pm
        self.amp_input = self.amp_input_0.copy()

        self.opd_internal = np.zeros_like(self.pm, dtype=float)

        # user override: if set to an array shaped like pupil mask, use static OPD input
        self.static_input_field: Optional[np.ndarray] = None

        self.frame_id = 0
        self._step_hook: Optional[Callable[[BaldrAppSimState], None]] = None

    def set_step_hook(self, fn: Optional[Callable[[BaldrAppSimState], None]]) -> None:
        self._step_hook = fn

    def _update_scint_amp(self) -> None:
        if not self.cfg.include_scintillation:
            self.amp_input = self.amp_input_0
            return

        # advance scint screen a bit
        for _ in range(int(self.cfg.jumps_per_iter)):
            self.scint_scrn.add_row()

        wavefront = np.exp(1j * self.scint_scrn.scrn)
        propagated = self.aotools.opticalpropagation.angularSpectrum(
            inputComplexAmp=wavefront,
            z=float(self.cfg.propagation_distance),
            wvl=float(self.zwfs_ns.optics.wvl0),
            inputSpacing=float(self.dx),
            outputSpacing=float(self.dx),
        )
        amp_scint = np.abs(propagated)
        self.amp_input = self.amp_input_0 * amp_scint

    def step(self) -> np.ndarray:
        """
        Updates turbulence/scint and returns opd_total = opd_input + opd_dm.
        """
        if self._step_hook is not None:
            self._step_hook(self)

        # OPD input from turbulence (AO1 residual) or static field
        if self.static_input_field is None:
            self.scrn.add_row()

            _, reco_1 = self.bldr.first_stage_ao(
                self.scrn,
                Nmodes_removed=int(self.cfg.Nmodes_removed),
                basis=self.basis,
                phase_scaling_factor=1.0,
                return_reconstructor=True,
            )
            self.reco_list.append(reco_1)
            ao_1 = self.basis[0] * (self.scrn.scrn - self.reco_list.pop(0))
            opd_input = (self.zwfs_ns.optics.wvl0 / (2.0 * np.pi)) * ao_1
        else:
            arr = np.asarray(self.static_input_field)
            if arr.shape != self.pm.shape:
                raise ValueError(f"static_input_field shape {arr.shape} != pupil mask shape {self.pm.shape}")
            opd_input = arr

        self._update_scint_amp()

        # DM contribution
        dm_cmd = np.asarray(getattr(self.zwfs_ns.dm, "current_cmd", np.zeros(140)), dtype=float).reshape(-1)
        opd_dm = self.bldr.get_dm_displacement(
            command_vector=dm_cmd,
            gain=self.zwfs_ns.dm.opd_per_cmd,
            sigma=self.zwfs_ns.grid.dm_coord.act_sigma_wavesp,
            X=self.zwfs_ns.grid.wave_coord.X,
            Y=self.zwfs_ns.grid.wave_coord.Y,
            x0=self.zwfs_ns.grid.dm_coord.act_x0_list_wavesp,
            y0=self.zwfs_ns.grid.dm_coord.act_y0_list_wavesp,
        )

        return opd_input + opd_dm


@dataclass
class SimCameraIO(CameraIO):
    def __init__(self, state: BaldrAppSimState, shape=(32, 32)):
        self.state = state
        self.shape = tuple(shape)


    @staticmethod
    def _crop_or_pad_center(im: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        # im: (H,W)
        Ht, Wt = shape
        H, W = im.shape

        # center-crop if too big
        if H > Ht:
            r0 = (H - Ht) // 2
            im = im[r0:r0 + Ht, :]
            H = Ht
        if W > Wt:
            c0 = (W - Wt) // 2
            im = im[:, c0:c0 + Wt]
            W = Wt

        # pad if too small
        if H < Ht or W < Wt:
            out = np.zeros((Ht, Wt), dtype=im.dtype)
            r0 = (Ht - H) // 2
            c0 = (Wt - W) // 2
            out[r0:r0 + H, c0:c0 + W] = im
            return out

        return im


    def get_frame(self, *, timeout_s: Optional[float] = None) -> Frame:
        _ = timeout_s  # unused
        opd_total = self.state.step()

        im = self.state.bldr.get_frame(
            opd_total,
            self.state.amp_input,
            self.state.opd_internal,
            self.state.zwfs_ns,
            detector=self.state.detector,
            use_pyZelda=self.state.cfg.use_pyZelda,
        ).astype(float)

        if im.ndim != 2:
            raise RuntimeError(f"SimCameraIO: expected 2D frame, got {im.shape}")

        im = self._crop_or_pad_center(im, self.shape)

        self.state.frame_id += 1
        return Frame(data=im, t_s=time.time(), frame_id=int(self.state.frame_id))

    def close(self) -> None:
        return None


@dataclass
class SimDMIO(DMIO):
    state: BaldrAppSimState

    def write(self, cmd: np.ndarray) -> None:
        cmd = np.asarray(cmd, dtype=float).reshape(-1)
        # keep whatever convention you want; this assumes cmd is your 140-vector
        if hasattr(self.state.zwfs_ns, "dm"):
            self.state.zwfs_ns.dm.current_cmd = cmd

    def close(self) -> None:
        return None
# # baldr_rtc/io/simulation_backend.py
# from __future__ import annotations

# import time
# from typing import Optional

# import numpy as np

# from .base import Frame, CameraIO, DMIO


# class SimCameraIO(CameraIO):
#     def __init__(self, frame: Optional[np.ndarray] = None):
#         if frame is None:
#             frame = np.zeros((256, 320), dtype=np.float32)
#         self._frame = np.asarray(frame, dtype=np.float32)
#         self._frame_id = 0

#     def get_frame(self, timeout_s: float | None = None) -> Frame:
#         fr = Frame(
#             data=self._frame,
#             t_s=time.time(),
#             frame_id=self._frame_id,
#         )
#         self._frame_id += 1
#         return fr


# class SimDMIO(DMIO):
#     def __init__(self, n_act: int = 140):
#         self.n_act = int(n_act)
#         self._last = np.zeros(self.n_act, dtype=np.float32)

#     def write(self, command: np.ndarray) -> None:
#         cmd = np.asarray(command, dtype=np.float32).reshape(-1)
#         if cmd.size != self.n_act:
#             raise ValueError(f"DM command must have length {self.n_act}, got {cmd.size}")
#         self._last = cmd

#     def read(self) -> np.ndarray:
#         return self._last


# def make_sim_io(
#     *,
#     frame: Optional[np.ndarray] = None,
#     n_act: int = 140,
# ) -> tuple[CameraIO, DMIO]:
#     cam = SimCameraIO(frame=frame)
#     dm = SimDMIO(n_act=n_act)
#     return cam, dm