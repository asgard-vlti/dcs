#!/usr/bin/env python
import os
from typing import Iterable, Iterator

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.axis import Axis
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from astropy.io import fits  # type: ignore
from matplotlib.animation import FuncAnimation

N_ACTX = 12
N_ACT = N_ACTX * N_ACTX


# def update(i: int, ax: Axes) -> Iterable[Artist]:
#     ax.images[0].set_data((imat @ eigvec[:, i]).reshape([17, 17]))
#     ax.images[0].autoscale()
#     ax.set_title(f"eigenmode {i}, eigenval: {eigval[i]}")
#     return [ax]

BALDR_ROOT = os.environ.get("BALDR_ROOT", "/usr/local/etc/")

if __name__ == "__main__":
    imat: NDArray = fits.getdata(BALDR_ROOT + "/B1_mode_to_meas.fits")  # type: ignore
    eigval, eigvec = np.linalg.eigh(imat.T @ imat + 1e-4 * np.eye(imat.shape[1]))
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]
    fig, axs = plt.subplots(12, 12, figsize=[12, 12])
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow((imat @ eigvec[:, i]).reshape([17, 17]))
        ax.set_title(f"{eigval[i]:0.2e}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("pca.png", dpi=200)
