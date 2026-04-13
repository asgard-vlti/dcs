from typing import Any, Callable, Optional
from dataclasses import dataclass, field
import inspect
import logging

import numpy as np
import zmq
import time

import Cam
import DM
import AO
import LazyPirateZMQ
import pickle
import pathlib
import datetime

# TODO: saving/loading of class and subclass from pickle
# TODO: ignore pix in BAO, needs pupil fitting and thinking about that, maybe fine tuning offload to detector
savepth = pathlib.Path("~/.config/minimal_baldr_rtc/").expanduser()
logger = logging.getLogger(__name__)


class BaldrAO:
    recon: Optional[AO.Reconstructor]
    controller: Optional[AO.Controller]

    def __init__(self, beam):
        self.beam = beam

        self.cam = Cam.Cam(beam)
        self.dm = DM.FourierDM(beam)
        self.L_max = float(getattr(self.dm, "L_max", 0.0))

        self.MDS = LazyPirateZMQ.ZmqLazyPirateClient(zmq.Context(), "tcp://mimir:5555")

        self.recon = None
        self.controller = None

        self.estimator = AO.StrehlEstimator(None, 0.0, 0.0)

        self.is_closed = False
        self.wants_to_close = False

        self.last_strehl_est = None

        self.iter = 0
        self.start_time = time.time()

    def run_iteration(self):
        img = self.cam.get_img()

        normed_img = self.cam.normalise(img).flatten()

        self.last_strehl_est = self.estimator.metric(normed_img)

        self.iter += 1
        if self.iter == 1000:
            elapsed = time.time() - self.start_time
            msg = f"FPS: {self.iter / elapsed:.2f}"
            self.iter = 0
            self.start_time = time.time()

            if self.recon is None:
                msg += " ... no recon"

            if self.controller is None:
                msg += " ... no controller"

            if np.all(np.abs(self.cam.dark) == 0.0):
                msg += " ... no dark"

            msg += (
                f" Close thresh: {self.estimator.close_threshold:.2e}, "
                f"open thresh: {self.estimator.open_threshold:.2e}"
            )
            logger.info(msg)

        if self.recon is None or self.controller is None or self.cam.dark is None:
            return

        normed_img = self.cam.normalise(img).flatten()
        self.last_strehl_est = self.estimator.metric(normed_img)

        if self.wants_to_close:
            if self.is_closed:
                if self.last_strehl_est < self.estimator.open_threshold:
                    logger.info(
                        "Estimator is %.2e (less than open thresh of %.2e)",
                        self.last_strehl_est,
                        self.estimator.open_threshold,
                    )
                    self.is_closed = False
                    self.dm.flatten()
                    self.controller.reset()
            else:
                if self.last_strehl_est > self.estimator.close_threshold:
                    logger.info(
                        "Estimator is %.2e (greater than close thresh of %.2e)",
                        self.last_strehl_est,
                        self.estimator.close_threshold,
                    )
                    self.is_closed = True
                else:
                    self.is_closed = False
                    logger.info(
                        "Estimator is %.2e (less than close thresh of %.2e)",
                        self.last_strehl_est,
                        self.estimator.close_threshold,
                    )

        if self.is_closed:
            # AO time
            error = self.recon.reconstruct(normed_img)
            command = self.controller.compute_command(error)
            self.dm.set_data(command)

    def set_open_threshold(self, new_thresh):
        self.estimator.open_threshold = float(new_thresh)

    def set_close_threshold(self, new_thresh):
        self.estimator.close_threshold = float(new_thresh)

    def servo(self, new_state: str):
        logger.info("in servo fn, %s", new_state)
        if new_state == "on":
            self.wants_to_close = True
            self.save_state("closing_loop")
        else:
            self.wants_to_close = False
            self.is_closed = False
            self.dm.flatten()
            if self.controller:
                self.controller.reset()

    def take_dark(self):
        cur_bmy = self.MDS.send_and_recv(f"read BMY{self.beam}")
        self.MDS.send_and_recv(f"moveabs BMY{self.beam} 500.0")
        time.sleep(3)
        self.cam.take_dark(256)
        self.MDS.send_and_recv(f"moveabs BMY{self.beam} {cur_bmy}")
        time.sleep(3)

    def take_pupil_img(self):
        res = self.MDS.send_and_recv(f"moverel BMX{self.beam} -200.0")
        logger.info("recieved %s from mds", res)
        time.sleep(3)
        pupil = self.cam.take_stack(256).mean(0)
        self.MDS.send_and_recv(f"moverel BMX{self.beam} 200.0")
        time.sleep(1)
        return pupil

    def update_estimator_mask(
        self, scattered_flux_mask_r_outer=12.0, scattered_flux_mask_r_inner=9.5
    ):
        logger.info("Taking pupil img")
        pupil_img = self.take_pupil_img()
        logger.info("Pupil image taken")
        self.estimator = AO.StrehlEstimator(
            mask=None, close_threshold=0.5, open_threshold=0.7
        )
        self.estimator.update_mask(
            pupil_img, scattered_flux_mask_r_outer, scattered_flux_mask_r_inner
        )

    @staticmethod
    def parse_block(ls):
        # the gains are going to be in blocks. Figure out the indicies and report
        # as e.g {"0:10": 0.1, "10:20": 0.2} rather than {"0": 0.1, "1": 0.1, ...}
        # each block can have variable length
        dict_out = {}
        block_start = 0
        for i in range(1, len(ls)):
            if ls[i] != ls[i - 1]:
                block_end = i
                dict_out[f"{block_start}:{block_end}"] = ls[i - 1]
                block_start = block_end
        dict_out[f"{block_start}:{len(ls)}"] = ls[-1]
        return dict_out

    def get_controller_params(self):
        """
        get the controller parameters, using blocks to simplify
        the amount of data transmitted
        """
        if self.controller is None:
            raise ValueError("Controller not created yet")

        gains = getattr(self.controller, "gains")
        leaks = getattr(self.controller, "leaks")
        return {
            "controller_type": type(self.controller).__name__,
            "gains": self.parse_block(gains),
            "leaks": self.parse_block(leaks),
        }

    def create_reconstructor(
        self,
        kind="Linear",
        ref_stack_nframes=1000,
        rcond=1e-3,
        amp=0.03,
        sleep=0.01,
        n_im=2,
        n_pokes=10,
        n_discard=1,
    ):
        ref = self.take_ref(ref_stack_nframes).flatten()
        # remove Lmax
        self.dm.L_max = 1.0
        logger.info("making new recon...")
        im = self.take_interaction_matrix(
            amp=amp,
            sleep=sleep,
            n_im=n_im,
            n_discard=n_discard,
            n_pokes=n_pokes,
        )
        logger.info("IM has shape %s", im.shape)
        self.dm.L_max = self.L_max

        if kind.lower() == "linear":
            self.recon = AO.LinearReconstructor(im, ref, rcond=rcond)
        elif kind.lower() == "PALinear":
            pupil_img = self.take_pupil_img()
            self.recon = AO.PupilAwareLinearReconstructor(
                im,
                pupil_img,
                self.cam,
                rcond=rcond,
            )

        logger.info("made new recon %s", self.recon)

    def update_reconstructor_pupil(self):
        if self.recon is None:
            raise ValueError("Reconstructor not created yet")

        if not isinstance(self.recon, AO.PupilAwareLinearReconstructor):
            raise ValueError(
                "Current reconstructor is not pupil aware, cannot update pupil"
            )

        sky_pupil_img = self.take_pupil_img()
        self.recon.update_reference(sky_pupil_img)

    def create_controller(self, type="leaky_integrator"):
        if type == "leaky_integrator":
            self.controller = AO.LeakyIntegrator(
                self.dm.n_acts,
                gains=np.full(self.dm.n_acts, 0.0, dtype=float),
                leaks=np.full(self.dm.n_acts, 0.99, dtype=float),
            )
        else:
            logger.error("Unknown controller type %s", type)
            raise ValueError(f"Unknown controller type {type}")

        logger.info("made new controller %s", self.controller)

    def _parse_indices(self, idxs):
        if isinstance(idxs, str):
            if ":" in idxs:
                start, stop = map(int, idxs.split(":"))
                return list(range(start, stop))
            return list(map(int, idxs.split(",")))
        if isinstance(idxs, (int, float, np.integer, np.floating)):
            return [int(idxs)]
        return list(idxs)

    def _parse_values(self, values):
        if isinstance(values, str):
            return list(map(float, values.split(",")))
        if isinstance(values, (int, float, np.integer, np.floating)):
            return [float(values)]
        return [float(v) for v in values]

    def _normalise_idx_value_inputs(self, idxs, values):
        idxs = self._parse_indices(idxs)
        values = self._parse_values(values)

        if len(values) == 1:
            values = values * len(idxs)
        elif len(values) != len(idxs):
            raise ValueError(
                f"Number of values ({len(values)}) must be 1 or match number of indices ({len(idxs)})"
            )

        return idxs, values

    def set_ki_gains(self, idxs, values):
        """
        Set the ki gains of the controller
        idxs can be a single number, comma separated list or a slice string like 0:10
        values can be a single number or a comma separated list of values to set the gains to
        """
        if self.controller is None:
            raise ValueError("Controller not created yet")

        idxs, values = self._normalise_idx_value_inputs(idxs, values)
        gains = getattr(self.controller, "gains")

        for idx, value in zip(idxs, values):
            gains[idx] = value
            logger.info("set gain of mode %s to %s", idx, value)

    def set_leaks(self, idxs, values):
        """
        Set the leak values of the controller
        idxs can be a single number, comma separated list or a slice string like 0:10
        values can be a single number or a comma separated list of values to set the leaks to
        """
        if self.controller is None:
            raise ValueError("Controller not created yet")

        idxs, values = self._normalise_idx_value_inputs(idxs, values)
        leaks = getattr(self.controller, "leaks")

        for idx, value in zip(idxs, values):
            leaks[idx] = value
            logger.info("set leak of mode %s to %s", idx, value)

    def take_interaction_matrix(
        self,
        amp,
        sleep=0.01,
        n_im=1,
        n_pokes=5,
        n_discard=2,
    ):
        n_modes = self.dm.n_acts
        responses = []

        for mode_idx in range(n_modes):
            res = 0.0
            for _ in range(n_pokes):
                imgs = []
                for sp in [-1, 1]:
                    cmd = np.zeros((n_modes, 1))
                    cmd[mode_idx] = sp * amp
                    self.dm.set_data(cmd)

                    time.sleep(sleep)

                    self.cam.take_stack(n_discard)
                    ims = self.cam.take_stack(n_im)

                    imgs.append(self.cam.normalise(ims).mean(0))
                res += (imgs[1] - imgs[0]) / (2 * amp * n_pokes)
            responses.append(res)

        self.dm.flatten()

        return np.array(responses).reshape(n_modes, -1)

    def take_ref(self, nframes=1000):
        self.dm.flatten()

        imgs = self.cam.take_stack(nframes)
        ref = imgs.mean(0)
        return self.cam.normalise(ref)

    def set_L_max(self, new_L_max):
        self.L_max = float(new_L_max)
        self.dm.L_max = self.L_max
        logger.info("Set L_max to %.3f", self.L_max)

    def save_state(self, filename):
        state = {
            "recon": self.recon,
            "controller": self.controller,
            "estimator": self.estimator,
            "cam_dark": self.cam.dark,
            "L_max": self.L_max,
            "desc": filename,
        }
        filename_with_time = (
            savepth
            / f"beam_{self.beam}"
            / f"bao_state_{filename}_{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}.pkl"
        )
        with open(filename_with_time, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filename):
        """
        Can take the filename (timestring excluded )as a string or an
        integer index to load the nth most recent state.
        """
        try:
            filename = int(filename)
        except ValueError:
            pass

        state_dir = savepth / f"beam_{self.beam}"

        if isinstance(filename, int):
            files = sorted(
                state_dir.glob("bao_state_*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not files:
                raise FileNotFoundError(f"No saved states found for beam {self.beam}")
            filename_with_time = files[filename]
        else:
            # Support either a full filename or a base description used by save_state.
            candidate = state_dir / filename
            if candidate.exists():
                filename_with_time = candidate
            else:
                fragment = str(filename)
                all_states = sorted(
                    state_dir.glob("bao_state_*.pkl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                matches = [
                    p for p in all_states if fragment in p.name[len("bao_state_") :]
                ]
                if not matches:
                    raise FileNotFoundError(
                        f"No saved state found for beam {self.beam} matching '{filename}'"
                    )
                filename_with_time = matches[0]

        with open(filename_with_time, "rb") as f:
            state = pickle.load(f)

        logger.info("read from %s", filename_with_time)
        logger.debug("loaded state: %s", state)

        self.recon = state["recon"]
        self.controller = state["controller"]
        self.estimator = state.get("estimator")
        if "L_max" in state:
            self.set_L_max(state["L_max"])
        else:
            self.L_max = float(getattr(self.dm, "L_max", self.L_max))
        if "cam_dark" in state:
            self.cam.dark = np.asarray(state["cam_dark"])

    def save_img_vs_ref(self):
        if self.recon is None:
            raise ValueError("Reconstructor not created yet")

        img = self.cam.get_img()
        normed_img = self.cam.normalise(img).flatten()
        ref = self.recon.ref

        diff = normed_img - ref

        save_dir = savepth / f"beam_{self.beam}" / "img_vs_ref"
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
        np.savez(
            save_dir / f"img_vs_ref_{timestamp}.npz",
            img=normed_img,
            ref=ref,
            diff=diff,
        )

    def get_status(self):
        status = {
            "servo": "on" if self.is_closed else "off",
            "wants_to_close": self.wants_to_close,
            "cnt": self.iter,
            "estimator_metric": (
                self.last_strehl_est
                if self.estimator
                else None if self.estimator else None
            ),
        }
        return status

    def get_settings(self):
        settings = {
            "open_threshold": self.estimator.open_threshold if self.estimator else None,
            "close_threshold": (
                self.estimator.close_threshold if self.estimator else None
            ),
            "L_max": self.L_max,
            "controller_type": (
                type(self.controller).__name__ if self.controller else None
            ),
            "controller_settings": (
                self.get_controller_params() if self.controller else None
            ),
            "reconstructor_type": type(self.recon).__name__ if self.recon else None,
        }
        return settings
