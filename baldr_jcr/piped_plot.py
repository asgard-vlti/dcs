from typing import Dict, Iterable

from matplotlib.artist import Artist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

if __name__ == "__main__":
    arr_len = 100
    arrs: Dict[str, NDArray] = {}
    arr_t = np.arange(arr_len)
    t = 0
    plottables = [
        "flux",
        "strehl_flux",
        "meas_cl_rms",
        # "mode_raw_rms",
        # "mode_filt_rms",
        # "com_raw_rms",
        "com_clean_rms",
    ]

    for plottable in plottables:
        arrs[plottable] = np.zeros(arr_len)
    n_sub = int(np.ceil(len(plottables) ** 0.5))
    fig, ax = plt.subplots(n_sub, n_sub, figsize=[10, 6])

    def update(idx):
        with open("/dev/stdin", "r") as f:
            while True:
                line = f.readline()
                if line == 0:
                    exit(0)
                skip = False
                if " ns " in line:
                    print(line)
                for plottable in plottables:
                    if f"|{plottable}=" not in line:
                        skip = True
                    break
                if skip:
                    continue
                break
        for plottable in plottables:
            arrs[plottable] = np.roll(arrs[plottable], -1)
            arrs[plottable][-1] = float(
                line.rsplit(f"|{plottable}=", 1)[1].split("|", 1)[0]
            )
        t = idx
        for i, plottable in enumerate(plottables):
            ax[i].lines[0].set_data(arr_t + t, arrs[plottable])
            ax[i].set_ylim(arrs[plottable].min(), arrs[plottable].max())
            ax[i].set_xlim(t, arr_len + t)
        return ax

    ax = ax.flatten()
    for i, plottable in enumerate(plottables):
        ax[i].plot(arr_t, arrs[plottable])
        ax[i].set_title(plottable)
        ax[i].set_xticks([])
    plt.tight_layout()
    anim = FuncAnimation(fig, update)
    plt.show()
