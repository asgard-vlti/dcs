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

    def get_img(self, subtract_dark=True):
        img = self.shm.get_data(check=self.last_cnt)
        self.last_cnt = self.shm.get_counter()

        if subtract_dark:
            return img - self.dark
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

    def take_dark(self, nframes):
        self.dark = self.take_stack(nframes, subtract_dark=False).mean(0)

    def normalise(self, stack):
        if stack.ndim == 2:
            return stack / np.sum(stack)
        return stack / np.sum(stack, axis=(-1, -2))[:, None, None]
