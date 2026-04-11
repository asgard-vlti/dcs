import numpy as np
import zmq
import time
import toml
import os
import argparse
import datetime
import glob

from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
from asgard_alignment import FLI_Cameras as FLI
import matplotlib.pyplot as plt

import basis_funcs
import consts


class DM:
    """wrapper of Frantz shm specifically for control of Asgard's DM's
    Adapted from dmclass in DM_shm_ctrl.py
    """

    def __init__(self, beam_id, main_chn=2, basis=None):
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
            print(f"added: {self.shmfs[ii]}")
        # actual combined shared memory
        if self.nch != 0:
            self.shm0 = shm(self.shmf0, nosem=False)
        else:
            print("Shared memory structures unavailable. DM server started?")

        if basis is not None:
            self.basis = basis
        else:
            self.basis = np.eye(consts.n_acts)

        self.n_acts = self.basis.shape[1]

    def set_data(self, cmd):
        """
        convention to apply any user specific commands on channel 2!
        """
        self.shms[self.main_chn].set_data(self.basis @ cmd)
        ##
        self.shm0.post_sems(1)

    def flatten(self):
        cmd = np.zeros(self.n_acts)
        for ii, ss in enumerate(self.shms):
            ss.set_data(cmd)

            print(f"zero'd {self.shmfs[ii]}")
        ##
        self.shm0.post_sems(1)


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
