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


class Cam:
    def __init__(self, beam):
        self.shm = shm(f"/dev/shm/baldr{beam}.im.shm", nosem=False)
        self.imsize = (self.shm.mtdata["x"], self.shm.mtdata["y"])
        self.dark = np.zeros((self.shm.mtdata["x"], self.shm.mtdata["y"]))

        self.nsem = 4
        self.shm.catch_up_with_sem(self.nsem)

        self.last_cnt = self.shm.get_counter()

        self.bad_pixels = []

    def get_img(self, subtract_dark=True):
        img = self.shm.get_data(check=self.last_cnt)
        self.last_cnt = self.shm.get_counter()

        if subtract_dark:
            img = img - self.dark

        # Correct bad pixels by replacing them with the mean of their 8-connected
        # neighbors. This is vectorized over the full frame instead of looping
        # over each bad pixel
        if len(self.bad_pixels) > 0:
            padded = np.pad(img, 1, mode="constant", constant_values=0)
            valid = np.pad(
                np.ones_like(img, dtype=np.float64),
                1,
                mode="constant",
                constant_values=0,
            )

            neighbor_sum = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, :-2]
                + padded[1:-1, 2:]
                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            )
            neighbor_count = (
                valid[:-2, :-2]
                + valid[:-2, 1:-1]
                + valid[:-2, 2:]
                + valid[1:-1, :-2]
                + valid[1:-1, 2:]
                + valid[2:, :-2]
                + valid[2:, 1:-1]
                + valid[2:, 2:]
            )

            bad_pixels = np.asarray(self.bad_pixels)
            xs = bad_pixels[:, 0]
            ys = bad_pixels[:, 1]
            img = img.copy()
            img[xs, ys] = neighbor_sum[xs, ys] / neighbor_count[xs, ys]

        return img

    def take_stack(self, nframes, subtract_dark=True):
        if nframes == 1:
            return self.get_img(subtract_dark)[None, :, :]

        imgs = np.zeros((nframes, *self.imsize))
        for i in range(nframes):
            imgs[i] = self.get_img(subtract_dark=False)
        if subtract_dark:
            imgs = imgs - self.dark[None, :, :]
        return imgs

    def take_dark(self, nframes, hot_pix_threshold=1100, stddev_threshold=100):
        dark_stack = self.take_stack(nframes, subtract_dark=False)
        self.dark = dark_stack.mean(axis=0)

        # identify bad pixels
        hot_pix = self.dark > hot_pix_threshold
        stddev_pix = dark_stack.std(axis=0) > stddev_threshold

        self.bad_pixels = np.argwhere(hot_pix | stddev_pix)
        print(f"Identified {len(self.bad_pixels)} bad pixels")
        for x, y in self.bad_pixels:
            print(
                f"  ({x}, {y}) with dark value {self.dark[x,y]:.1f} and stddev {dark_stack[:,x,y].std():.1f}"
            )

    def normalise(self, stack):
        if stack.ndim == 2:
            return stack / np.sum(stack)
        return stack / np.sum(stack, axis=(-1, -2))[:, None, None]
