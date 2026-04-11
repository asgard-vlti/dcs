from typing import Any, Callable, Optional
from dataclasses import dataclass, field
import inspect

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


class BaldrAO:
    recon: Optional[AO.Reconstructor]
    controller: Optional[AO.Controller]

    def __init__(self, beam):
        self.beam = beam

        self.cam = Cam.Cam(beam)
        self.dm = DM.FourierDM(beam)

        self.MDS = LazyPirateZMQ.ZmqLazyPirateClient(zmq.Context(), "tcp://mimir:5555")

        self.recon = None
        self.controller = None

        self.is_closed = False

        self.iter = 0
        self.start_time = time.time()

    def run_iteration(self):
        img = self.cam.get_img()

        self.iter += 1
        if self.iter == 1000:
            elapsed = time.time() - self.start_time
            print(f"\rFPS: {self.iter / elapsed:.2f}", end="")
            self.iter = 0
            self.start_time = time.time()

            if self.recon is None:
                print(" ... no recon", end="")

            if self.controller is None:
                print(" ... no controller", end="")

        if self.recon is None or self.controller is None:
            return

        if self.is_closed:
            # AO time
            normed_img = self.cam.normalise(img).flatten()
            error = self.recon.reconstruct(normed_img)
            command = self.controller.compute_command(error)
            self.dm.set_data(command)

    def servo(self, new_state: str):
        # Future improvement: add a lock-state estimator before enabling closed loop.
        if new_state == "on":
            self.is_closed = True
        else:
            self.is_closed = False

    def take_dark(self):
        # TODO: change to use BMY instead of shutters
        # self.MDS.send_and_recv(f"b_shut close {self.beam}")
        cur_bmy = self.MDS.send_and_recv(f"read BMY{self.beam}")
        self.MDS.send_and_recv(f"moveabs BMY{self.beam} 1000.0")
        time.sleep(3)
        self.cam.take_dark(256)
        # self.MDS.send_and_recv(f"b_shut open {self.beam}")
        self.MDS.send_and_recv(f"moveabs BMY{self.beam} {cur_bmy}")
        time.sleep(3)

    def take_pupil_img(self):
        self.MDS.send_and_recv(f"movrel BMX{self.beam} -200.0")
        time.sleep(3)
        pupil = self.cam.take_stack(256)
        self.MDS.send_and_recv(f"movrel BMX{self.beam} 200.0")
        return pupil

    def create_reconstructor(self, ref_stack_nframes=1000, rcond=1e-3):
        ref = self.take_ref(ref_stack_nframes).flatten()
        print(f"\n making new recon...")
        im = self.take_interaction_matrix(amp=0.02, n_im=10, n_pokes=1, n_discard=2)
        print(f"\n IM has shape {im.shape}")
        self.recon = AO.LinearReconstructor(im, ref, rcond=rcond)

        print(f"\n made new recon {self.recon}")

    def create_controller(self):
        self.controller = AO.LeakyIntegrator(self.dm.n_acts, gains=0.0, leaks=0.9)
        print(f"\n made new controller {self.controller}")

    def set_ki_gains(self, idxs, values):
        """
        Set the ki gains of the controller
        idxs can be a single number, comma separated list or a slice string like 0:10
        values can be a single number or a comma separated list of values to set the gains to
        """
        if self.controller is None:
            raise ValueError("Controller not created yet")

        if isinstance(idxs, str):
            if ":" in idxs:
                start, stop = map(int, idxs.split(":"))
                idxs = range(start, stop)
            else:
                idxs = list(map(int, idxs.split(",")))

        if isinstance(values, str):
            values = list(map(float, values.split(",")))
            if len(values) == 1:
                values = values * len(idxs)

        for idx, value in zip(idxs, values):
            self.controller.gains[idx] = value

            print(f"\n set gain of mode {idx} to {value}")

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

    def save_state(self, filename):
        state = {
            "recon": self.recon,
            "controller": self.controller,
        }
        filename_with_time = (
            savepth
            / f"beam_{self.beam}"
            / f"bao_state_{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}.pkl"
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

        if isinstance(filename, int):
            files = sorted(
                (savepth / f"beam_{self.beam}").glob("bao_state_*.pkl"), reverse=True
            )
            if not files:
                raise FileNotFoundError(f"No saved states found for beam {self.beam}")
            filename_with_time = files[filename]
        else:
            filename_with_time = savepth / f"beam_{self.beam}" / filename

        with open(filename_with_time, "rb") as f:
            state = pickle.load(f)

        print(f"\nread from {filename_with_time}")
        print(state)

        self.recon = state["recon"]
        self.controller = state["controller"]

    def get_status(self):
        status = {
            "servo": "on" if self.is_closed else "off",
            "cnt": self.iter,
        }
        return status
