from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from baldr_python_rtc.baldr_rtc.io.simulation_backend import (
    BaldrAppSimConfig,
    BaldrAppSimState,
    SimCameraIO,
    SimDMIO,
)


def run_simulation_smoketest(
    n_steps: int = 80,
    poke_amp: float = 0.02,
    poke_act: int = 60,
    poke_period: int = 8,
    out_png: str = "sim_backend_verification.png",
) -> str:
    """
    Minimal functional test for io.mode='simulation'.

    - Instantiates default BaldrAppSimState (hard-coded default config inside simulation_backend.py)
    - Wraps it in SimCameraIO / SimDMIO
    - Steps a few frames
    - Applies a small periodic DM "poke" to verify the DM write path affects output
    - Saves a verification plot
    """

    cfg = BaldrAppSimConfig(use_pyZelda=bool(os.environ.get("BALDR_SIM_USE_PYZELDA", "0") == "1"))
    state = BaldrAppSimState(cfg=cfg)
    cam = SimCameraIO(state=state)
    dm = SimDMIO(state=state)

    # Collect metrics + some frames
    means = np.zeros(n_steps, dtype=float)
    stds = np.zeros(n_steps, dtype=float)
    frame_ids = np.zeros(n_steps, dtype=int)

    first = None
    last = None

    # Base DM command
    cmd = np.zeros(140, dtype=float)

    for k in range(n_steps):
        # Periodic dither on one actuator to prove DM->image coupling works
        if poke_period > 0 and (k % poke_period == 0):
            cmd[:] = 0.0
            if 0 <= poke_act < cmd.size:
                cmd[poke_act] = poke_amp
            dm.write(cmd)
        elif poke_period > 0 and (k % poke_period == poke_period // 2):
            # flip sign halfway through the period
            cmd[:] = 0.0
            if 0 <= poke_act < cmd.size:
                cmd[poke_act] = -poke_amp
            dm.write(cmd)

        fr = cam.get_frame()
        img = np.asarray(fr.data, dtype=float)

        if first is None:
            first = img.copy()
        last = img.copy()

        means[k] = float(np.mean(img))
        stds[k] = float(np.std(img))
        frame_ids[k] = int(fr.frame_id)

    # --- Plot ---
    fig = plt.figure(figsize=(11, 7))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(frame_ids, means)
    ax1.set_title("Mean intensity vs frame_id")
    ax1.set_xlabel("frame_id")
    ax1.set_ylabel("mean(I)")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(frame_ids, stds)
    ax2.set_title("Std intensity vs frame_id")
    ax2.set_xlabel("frame_id")
    ax2.set_ylabel("std(I)")

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(first, origin="lower")
    ax3.set_title("First frame")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 2, 4)
    diff = last - first
    im4 = ax4.imshow(diff, origin="lower")
    ax4.set_title("Last - First")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"simulation_backend smoketest | n_steps={n_steps}, poke_act={poke_act}, poke_amp={poke_amp}",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return out_png


if __name__ == "__main__":
    out = run_simulation_smoketest()
    print(f"[OK] Wrote verification plot: {out}")