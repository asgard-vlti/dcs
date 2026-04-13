import numpy as np
import zmq
import time
import toml
import os
import argparse
import datetime
import glob
import logging

from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
from asgard_alignment import FLI_Cameras as FLI
import matplotlib.pyplot as plt

import basis_funcs
import consts

logger = logging.getLogger(__name__)


class DM:
    """wrapper of Frantz shm specifically for control of Asgard's DM's
    Adapted from dmclass in DM_shm_ctrl.py
    """

    def __init__(self, beam_id, main_chn=2, basis=None, piston_free=True):
        beam_id = int(beam_id)

        # beam number
        self.beam_id = beam_id
        # sub channels shared memory
        self.shmfs = np.sort(glob.glob(f"/dev/shm/dm{beam_id}disp*.im.shm"))
        # combined channels
        self.shmf0 = f"/dev/shm/dm{beam_id}.im.shm"
        # number of sub channels
        self.nch = len(self.shmfs)
        # main channel to apply DM commands to
        self.main_chn = main_chn
        # actual shared memory objects
        self.shms = []
        for ii in range(self.nch):
            self.shms.append(shm(self.shmfs[ii], nosem=False))
            logger.info("added: %s", self.shmfs[ii])
        # actual combined shared memory
        if self.nch != 0:
            self.shm0 = shm(self.shmf0, nosem=False)
        else:
            logger.warning("Shared memory structures unavailable. DM server started?")

        if basis is not None:
            self.basis = basis
        else:
            self.basis = np.eye(consts.n_acts)

        self.n_acts = self.basis.shape[1]

        self.piston_free = piston_free

        self.L_max = 0.11

    def set_data(self, cmd):
        """
        convention to apply any user specific commands on channel 2!
        """
        cmd = self.basis @ cmd
        if self.piston_free:
            cmd = cmd - cmd.mean()

        if self.L_max < 0.1:
            logging.info("Applying Laplacian limiter with L_max = %.3f", self.L_max)
            cmd = self.laplacian_limiter(
                cmd.reshape(consts.act_shape), self.L_max
            ).flatten()

        self.shms[self.main_chn].set_data(cmd)
        ##
        self.shm0.post_sems(1)

    def flatten(self):
        self.set_data(np.zeros(self.n_acts))

    @staticmethod
    def laplacian_limiter(surface, L_max, return_L=False):
        # surface is a 2D array of actuator values
        # we will modify surface in place
        new_surface = surface.copy()

        laplacian = (
            -4 * surface[1:-1, 1:-1]
            + surface[1:-1, 2:]
            + surface[1:-1, :-2]
            + surface[2:, 1:-1]
            + surface[:-2, 1:-1]
        )

        new_surface[1:-1, 1:-1] += np.clip((laplacian - L_max) / 4, 0, None)
        # and the opposite for negative Laplacian
        new_surface[1:-1, 1:-1] += np.clip((laplacian + L_max) / 4, None, 0)

        # retain pinning
        new_surface[0, 1:-1] = new_surface[1, 1:-1]
        new_surface[-1, 1:-1] = new_surface[-2, 1:-1]
        new_surface[1:-1, 0] = new_surface[1:-1, 1]
        new_surface[1:-1, -1] = new_surface[1:-1, -2]

        if return_L:
            L_values = np.zeros_like(surface)
            L_values[1:-1, 1:-1] = laplacian
            return new_surface, L_values
        return new_surface


class FourierDM(DM):
    def __init__(
        self,
        beam_id,
        main_chn=2,
        min_freq_HO=1.1,
        max_freq_HO=5.01,
        spacing_HO=1.0,
        start_HO=0.0,
        orthogonalise=False,
        pin_edges=True,
    ):
        basis, _ = basis_funcs.fourier_basis(
            basis_funcs.make_hc_act_grid(),
            min_freq_HO,
            max_freq_HO,
            spacing_HO,
            start_HO,
            orthogonalise,
            pin_edges,
        )
        basis = basis.transformation_matrix
        super().__init__(beam_id, main_chn, basis)
