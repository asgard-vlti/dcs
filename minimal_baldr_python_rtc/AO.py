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

import pathlib
import scipy.optimize as opt
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
from asgard_alignment import FLI_Cameras as FLI
import matplotlib.pyplot as plt

import utils
import basis_funcs


class Reconstructor:
    @abc.abstractmethod
    def reconstruct(self, normed_img):
        pass


class LinearReconstructor(Reconstructor):
    def __init__(self, IM, ref, rcond=1e-3):
        self.recon_matrix = hcipy.inverse_tikhonov(IM.T, rcond=rcond)
        self.ref = ref

    def reconstruct(self, normed_img):
        return self.recon_matrix @ (normed_img - self.ref)


class Controller(abc.ABC):
    @abc.abstractmethod
    def compute_command(self, error):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class LeakyIntegrator(Controller):
    def __init__(self, n, gains=None, leaks=None):
        self.n = n
        if gains is None:
            self.gains = np.ones(n) * 0.1
        else:
            self.gains = gains
        if leaks is None:
            self.leaks = np.ones(n) * 0.99
        else:
            self.leaks = leaks

        self.command = np.zeros(n)

    def compute_command(self, error):
        self.command = self.leaks * self.command - self.gains * error
        return self.command

    def reset(self):
        self.command = np.zeros(self.n)


class StrehlEstimator:
    def __init__(self, mask, close_threshold, open_threshold):
        self.mask = mask
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold

    def update_mask(self, pupil_img, scattered_flux_mask_r_outer=12, scattered_flux_mask_r_inner = 9.5):
        cam_grid = hcipy.make_pupil_grid(32, diameter=32)

        res = opt.minimize(
            utils.xcor_sum_model,
            x0=[8, 0, 0],
            args=((pupil_img, cam_grid, 0.5),),
            bounds=((8, 8), (-10, 10), (-10, 10)),
        )
        img_center = np.array([15.5, 15.5])
        pupil_center = np.array([res.x[1], res.x[2]]) + img_center

        scattered_flux_mask = (
            utils.smooth_circle(
                cam_grid,
                scattered_flux_mask_r_outer,
                centre=pupil_center - img_center,
                softening=0.01,
            )
            - utils.smooth_circle(
                cam_grid,
                scattered_flux_mask_r_inner,
                centre=pupil_center - img_center,
                softening=0.01,
            )
        ).reshape(cam_grid.shape)

        self.mask = scattered_flux_mask > 0.5

        plt.figure()
        plt.imshow(pupil_img)
        plt.contour(scattered_flux_mask, levels=[0.5], colors="r")
        plt.contour(scattered_flux_mask, ":", levels=[0.1], colors="w")
        plt.scatter(pupil_center[0], pupil_center[1], c="r")
        plt.title("Pupil image with scattered flux mask")
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pth = pathlib.Path(f"~/tmp/baldr_minimal_py/pupil_with_mask_{dt}.png").expanduser()
        plt.savefig(pth)

    def should_close(self, normed_img):
        return self.metric(normed_img) < self.close_threshold

    def should_open(self, normed_img):
        return self.metric(normed_img) > self.open_threshold

    def metric(self, normed_img):
        if self.mask is None:
            return 0.0
        masked_img = normed_img[self.mask.flatten()]
        return np.median(masked_img)

    def set_open_threshold(self, new_threshold):
        self.open_threshold = new_threshold
        print(f"Close: {self.close_threshold}, open: {self.open_threshold}")

    def set_close_threshold(self, new_threshold):
        self.close_threshold = new_threshold
        print(f"Close: {self.close_threshold}, open: {self.open_threshold}")
