import numpy as np
import asgard_alignment.DM_modes2 as DM_modes2
import minimal_baldr_python_rtc.model as model
import zmq
import time
import toml
import os
import argparse
import datetime
import subprocess

from asgard_alignment.bcam import Bcam

from astropy.io import fits
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hcipy
from asgard_alignment import FLI_Cameras as FLI
import scipy.optimize as opt

parser = argparse.ArgumentParser(
    description="Flatten beam wavefront using hardware in the loop optimization."
)
parser.add_argument("beam", type=int, help="Beam number")

parser.add_argument(
    "--target",
    choices=["stddev", "model"],
    default=False,
    help="Whether the target should be visual flatness or a model based reference. "
    "Each is saved into its own flat file at the end",
)
parser.add_argument(
    "--show-plots",
    action="store_true",
    default=False,
    help="Show plots at the end of optimization",
)
args = parser.parse_args()

TARGET_TO_FLAT_NAME = {
    "stddev": "night-standard",
    "model": "test",
}


def main():
    beam = args.beam
    show_plots = args.show_plots

    def mds_connect(host: str, port: int = 5555, timeout_ms: int = 5000):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.connect(f"tcp://{host}:{port}")
        return ctx, sock

    def mds_send(sock, msg: str) -> str:
        sock.send_string(msg)
        return sock.recv_string().strip()

    ctx, sock = mds_connect("mimir")

    dm = dmclass(beam)

    cam = Bcam(beam)

    print(f"")

    cur_bmy = mds_send(sock, f"read BMY{beam}")
    mds_send(sock, f"moveabs BMY{beam} 500.0")
    time.sleep(3)
    cam.take_dark(256)
    if show_plots:
        plt.imshow(cam.dark)
        plt.colorbar()
        plt.show()

    mds_send(sock, f"moveabs BMY{beam} {cur_bmy}")
    time.sleep(3)

    print(f"Taking pupil only image for beam {beam}...")
    offset = 200.0
    mds_send(sock, f"moverel BMX{beam} {offset}")
    mds_send(sock, f"moverel BMY{beam} {offset}")
    time.sleep(1)

    pupil_only = cam.take_stack(1000).mean(0)

    mds_send(sock, f"moverel BMX{beam} {-offset}")
    mds_send(sock, f"moverel BMY{beam} {-offset}")
    time.sleep(1)

    if show_plots:
        plt.imshow(pupil_only)
        plt.colorbar()
        plt.show()

    act_grid = DM_modes2.make_hc_act_grid()
    fourier, freqs_used = DM_modes2.fourier_basis(
        act_grid,
        min_freq_HO=1.1,
        max_freq_HO=5.01,
        spacing_HO=1.0,
        start_HO=0.0,
        orthogonalise=False,
        pin_edges=True,
    )

    cam_grid = hcipy.make_pupil_grid(32, diameter=32)

    def smooth_circle(grid, radius, softening=0.1, centre=(0, 0)):
        r = np.sqrt((grid.x - centre[0]) ** 2 + (grid.y - centre[1]) ** 2)
        return 1 / (1 + np.exp((r - radius) / softening))

    def xcor_sum_model(params, args):
        img, grid, softening = args
        img /= np.sum(img)
        model = smooth_circle(
            grid, radius=params[0], softening=softening, centre=(params[1], params[2])
        ).reshape(grid.shape)
        model /= model.sum()
        return -np.sum(img * model)

    def xcor_sum(params, args):
        (img,) = args
        img /= np.sum(img)

    res = opt.minimize(
        xcor_sum_model,
        x0=[8, 0, 0],
        args=((pupil_only, cam_grid, 0.5),),
        bounds=((8, 8), (-10, 10), (-10, 10)),
    )

    pupil_mask = smooth_circle(
        cam_grid, radius=res.x[0], softening=0.5, centre=(res.x[1], res.x[2])
    ).reshape(32, 32)
    pupil_center = (res.x[1], res.x[2])

    if show_plots:
        plt.imshow(pupil_only)
        plt.contour(pupil_mask, levels=[0.5], colors="r")
        plt.show()

    # pupil_mask =
    scattered_flux_mask_r_outer = 12
    scattered_flux_mask_r_inner = 9.5
    scattered_flux_mask = (
        smooth_circle(
            cam_grid, scattered_flux_mask_r_outer, centre=pupil_center, softening=0.01
        )
        - smooth_circle(
            cam_grid, scattered_flux_mask_r_inner, centre=pupil_center, softening=0.01
        )
    ).reshape(cam_grid.shape)

    # plt.imshow(scattered_flux_mask)
    if show_plots:
        plt.imshow(pupil_only)
        plt.contour(scattered_flux_mask, levels=[0.5], colors="r")
        plt.contour(scattered_flux_mask, ":", levels=[0.1], colors="w")
        plt.show()

    def flux_outside_pupil(img, scatter_mask):
        return np.sum(img * scatter_mask)

    def uniformity_in_pupil(img, pupil_mask):
        img_in_pupil = img * pupil_mask
        mean_in_pupil = np.sum(img_in_pupil) / np.sum(pupil_mask)
        # want a uniform distribution in the pupil, so penalise the variance
        return np.sqrt(np.sum(pupil_mask * (img_in_pupil - mean_in_pupil) ** 2))

    def stddev_loss(cmd, lamb_unif, scatter_mask, pupil_mask):
        dm.set_data(cmd)
        time.sleep(0.01)
        img = cam.take_stack(64).mean(0)

        f = flux_outside_pupil(img, scatter_mask=scatter_mask)
        u = uniformity_in_pupil(img, pupil_mask=pupil_mask)
        l = float(-f + lamb_unif * u)
        print(np.sqrt(np.mean(cmd**2)), f"{l:.3f}")
        return l

    init_cmd = np.zeros(144)
    scattered_flux_mask /= scattered_flux_mask.sum()

    dm.set_data(init_cmd)
    time.sleep(0.01)
    img = cam.take_stack(64).mean(0)

    pupil_hard_mask = pupil_mask > 0.6

    print(stddev_loss(init_cmd, 0.1, scattered_flux_mask, pupil_mask))
    print(
        stddev_loss(np.random.randn(144) * 0.02, 0.1, scattered_flux_mask, pupil_mask)
    )

    time.sleep(1)

    def basis_loss(coeffs, basis, lamb_unif, scatter_mask, pupil_mask, scale=0.05):
        coeffs_scaled = coeffs * scale
        cmd = basis.linear_combination(coeffs_scaled)
        return stddev_loss(cmd, lamb_unif, scatter_mask, pupil_mask)

    def model_loss(coeffs, basis, model_img, pupil_mask, scale=0.05):
        coeffs_scaled = coeffs * scale
        cmd = basis.linear_combination(coeffs_scaled)
        dm.set_data(cmd)
        time.sleep(0.01)
        img = cam.take_stack(64).mean(0)

        img_in_pupil = img * pupil_mask
        model_in_pupil = model_img * pupil_mask

        img_in_pupil /= np.sum(img_in_pupil)
        model_in_pupil /= np.sum(model_in_pupil)

        return -np.sum(img_in_pupil * model_in_pupil)
        

    if args.target == "stddev":
        loss = basis_loss
        loss_args = (0.3, scattered_flux_mask, pupil_mask, 0.1)
    elif args.target == "model":


    freqs = [2.01, 3.51, 5.01]
    n_iters = [50, 120, 240]

    init_coeffs = None

    for freq, n_it in zip(freqs, n_iters):
        fourier = DM_modes2.fourier_basis(
            act_grid,
            min_freq_HO=1.1,
            max_freq_HO=freq,
            spacing_HO=1.0,
            start_HO=0.0,
            orthogonalise=True,
            pin_edges=True,
        )[0]

        n_modes = fourier.num_modes

        if init_coeffs is None:
            init_coeffs = np.zeros(n_modes)
        else:
            init_coeffs = fourier.coefficients_for(
                prev_fourier.linear_combination(init_coeffs)
            )

        res = opt.minimize(
            basis_loss,
            init_coeffs,
            (fourier, 0.3, scattered_flux_mask, pupil_mask, 0.1),
            method="COBYLA",
            options={"disp": True, "maxiter": n_it},
            # bounds=[[-0.05, 0.05] for _ in range(n_modes)],
        )

        print(f"Loss at end of optimization with {n_modes} modes: {res.fun:.3f}")

        prev_fourier = fourier

    # Apply final optimization result to DM
    final_coeffs = res.x * 0.1  # Apply the final scale factor
    final_cmd = fourier.linear_combination(final_coeffs)
    dm.set_data(final_cmd)
    time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"Optimization complete. Result set on DM{beam}.")
    print(f"RMS command: {np.sqrt(np.mean(final_cmd**2)):.6f}")
    print("=" * 60)
    print("\nVisually inspect the beam in the camera.")
    confirm = input("Does the beam look good? (yes/no): ").strip().lower()

    if confirm in ["yes", "y"]:
        print(f"\nApplying optimization as night standard for beam {beam}...")

        # Save current state
        subprocess.run(["flat-save", str(beam), "night-standard"], check=True)
        print(f"Saved flat to flat-save {beam} night-standard")

        # Flatten the DM
        dm.set_data(np.zeros(144))
        time.sleep(0.5)
        print("DM flattened")

        # Load the standard
        subprocess.run(["flat-load", str(beam), "night-standard"], check=True)
        print(f"Loaded night standard for beam {beam}")

        print("\nFlattening complete!")
    else:
        print("Optimization not applied. Clearing DM...")
        dm.set_data(np.zeros(144))
        time.sleep(0.5)


if __name__ == "__main__":
    main()
