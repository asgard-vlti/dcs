import numpy as np
import zmq
import time
import toml
import os
import argparse
import datetime
import glob
import hcipy
import abc

from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
from asgard_alignment import FLI_Cameras as FLI
import matplotlib.pyplot as plt

import basis_funcs


class Reconstructor:
    @abc.abstractmethod
    def reconstruct(self, normed_img):
        pass


class LinearReconstructor(Reconstructor):
    def __init__(self, IM, ref, rcond=1e-3):
        self.recon_matrix = hcipy.inverse_tikhonov(IM, rcond=rcond)
        self.ref = ref

    def reconstruct(self, normed_img):
        return self.recon_matrix @ (normed_img - self.ref)


class Controller(abc.ABC):
    @abc.abstractmethod
    def compute_command(self, error):
        pass


class LeakyIntegrator(Controller):
    def __init__(self, n, gains=None, leaks=None):
        self.n = n
        if gains is None:
            self.gains = np.ones(n) * 0.1
        else:
            self.gains = gains
        if leaks is None:
            self.leaks = np.ones(n) * 0.9
        else:
            self.leaks = leaks

        self.command = np.zeros(n)

    def compute_command(self, error):
        self.command = self.leaks * self.command + self.gains * error
        return self.command


class StrehlEstimator:
    def __init__(self, mask, close_threshold, open_threshold):
        self.mask = mask
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold

    def update_mask(self, pupil_img):
        

    def should_close(self, normed_img):
        masked_img = normed_img[self.mask]
        return masked_img.sum() < self.close_threshold

    def should_open(self, normed_img):
        masked_img = normed_img[self.mask]
        return masked_img.sum() > self.open_threshold
