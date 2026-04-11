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


# TODO: saving/loading of class and subclass from pickle
# TODO: ignore pix in BAO, needs pupil fitting and thinking about that, maybe fine tuning offload to detector

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
        self.iter += 1
        if self.iter == 1000:
            elapsed = time.time() - self.start_time
            print(f"FPS: {self.iter / elapsed:.2f}")
            self.iter = 0
            self.start_time = time.time()

        if self.recon is None or self.controller is None:
            return

        if self.is_closed:
            # AO time
            img = self.cam.get_img()
            normed_img = self.cam.normalise(img)
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
        self.MDS.send_and_recv(f"b_shut close {self.beam}")
        time.sleep(3)
        self.cam.take_dark(256)
        self.MDS.send_and_recv(f"b_shut open {self.beam}")

    def take_pupil_img(self):
        self.MDS.send_and_recv(f"movrel BMX{self.beam} -200.0")
        time.sleep(3)
        pupil = self.cam.take_stack(256)
        self.MDS.send_and_recv(f"movrel BMX{self.beam} 200.0")
        return pupil


    def create_reconstructor(self, ref_stack_nframes=1000, rcond=1e-3):
        ref = self.take_ref(ref_stack_nframes)
        im = self.take_interaction_matrix(amp=0.02, n_im=10, n_pokes=5, n_discard=2)
        self.recon = AO.LinearReconstructor(im, ref, rcond=rcond)

    def create_controller(self):
        self.controller = AO.LeakyIntegrator(self.dm.n_acts, gains=0.0, leaks=0.9)

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

        return np.array(responses)

    def take_ref(self, nframes=1000):
        self.dm.flatten()

        imgs = self.cam.take_stack(nframes)
        ref = imgs.mean(0)
        return self.cam.normalise(ref)
