#!/usr/bin/env python
import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from astropy.io import fits  # type: ignore

N_ACTX = 12
N_ACT = N_ACTX * N_ACTX

BALDR_ROOT = os.environ.get("BALDR_ROOT", "/usr/local/etc/")
BEAM = int(os.environ.get("BEAM", 1))

PLOT_LIM = int(os.environ.get("PLOT_LIM", 8))

if __name__ == "__main__":
    imat: NDArray = fits.getdata(BALDR_ROOT + f"/B{BEAM}_mode_to_meas.fits")  # type: ignore
    n_meas, n_mode = imat.shape
    n_mode_sqrt: int = int(np.ceil(n_mode**0.5))
    n_meas_sqrt: int = int(np.ceil(n_meas**0.5))
    eigval, eigvec = np.linalg.eigh(imat.T @ imat + 1e-4 * np.eye(imat.shape[1]))
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]
    if PLOT_LIM > 0:
        n_plot_sqrt = min(PLOT_LIM, n_meas_sqrt)
    else:
        n_plot_sqrt = n_meas_sqrt
    fig, axs = plt.subplots(n_plot_sqrt, n_plot_sqrt, figsize=[12, 12])
    axs = axs.flatten()
    for i in range(n_plot_sqrt**2):
        ax = axs[i]
        ax.imshow((imat @ eigvec[:, i]).reshape([n_meas_sqrt, n_meas_sqrt]))
        ax.set_title(f"{eigval[i]:0.2e}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("pca.png", dpi=200)
