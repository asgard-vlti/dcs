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


# units are: depth [fraction of pi], diameter [microns]
phasemask_parameters = {
    # "J1": {"depth": 0.5, "diameter": 54},
    # "J2": {"depth": 0.5, "diameter": 44},
    # "J3": {"depth": 0.5, "diameter": 36},
    # "J4": {"depth": 0.5, "diameter": 32},
    # "J5": {"depth": 0.5, "diameter": 65},
    "H5": {"depth": 0.5, "diameter": 68},
    "H4": {"depth": 0.5, "diameter": 53},
    "H3": {"depth": 0.5, "diameter": 44},
    "H2": {"depth": 0.5, "diameter": 37},
    "H1": {"depth": 0.5, "diameter": 31},
}


def generate_zwfs_model_image(
    case,  # one of ["AT", "UT", "Lab"]
    phasemask,  # one of ["J1-5", "H1-5"]
    centre,
    include_cold_stop=True,
    n_pix_pupil=256,
    n_pix_final=32,
):
    # validate all inputs
    if case not in ["AT", "UT", "Lab"]:
        raise ValueError(f"Invalid case: {case}. Must be one of ['AT', 'UT', 'Lab']")
    if phasemask not in phasemask_parameters.keys():
        raise ValueError(
            f"Invalid phasemask: {phasemask}. Must be one of {phasemask_parameters.keys()}"
        )

    if phasemask.startswith("J"):
        wavelength_wfs = 1.25e-6
    elif phasemask.startswith("H"):
        wavelength_wfs = 1.65e-6

    phasemask_diam = phasemask_parameters[phasemask]["diameter"] * 1e-6
    phasemask_depth = phasemask_parameters[phasemask]["depth"]

    lab_diam = 12e-3

    if case == "AT":
        telescope_diameter = 1.8
        secondary_diameter = 0.14
        aperture = hcipy.make_obstructed_circular_aperture(
            pupil_diameter=telescope_diameter,
            central_obscuration_ratio=secondary_diameter / telescope_diameter,
            num_spiders=4,
            spider_width=0.01,
        )
    if case == "UT":
        telescope_diameter = 8.2
        secondary_diameter = 1.1
        aperture = hcipy.make_obstructed_circular_aperture(
            pupil_diameter=telescope_diameter,
            central_obscuration_ratio=secondary_diameter / telescope_diameter,
            num_spiders=4,
            spider_width=0.01,
        )
    if case == "Lab":
        telescope_diameter = 8.2
        secondary_diameter = 1.1
        aperture = hcipy.make_obstructed_circular_aperture(
            pupil_diameter=telescope_diameter,
            central_obscuration_ratio=secondary_diameter / telescope_diameter,
            num_spiders=0,
        )

    # convert centre from pixels to physical units
    centre = centre.copy()
    centre -= np.array([(n_pix_final - 1) / 2, (n_pix_final - 1) / 2])
    centre = 2 * centre * telescope_diameter / n_pix_final

    pupil_grid = hcipy.make_pupil_grid(n_pix_pupil, 2 * telescope_diameter)
    pupil_grid = pupil_grid.shift(-centre)
    pupil = hcipy.evaluate_supersampled(aperture, pupil_grid, 6)

    # hcipy.imshow_field(pupil)

    focal_length = 254e-3
    loD = wavelength_wfs / lab_diam

    phase_mask_diam_loD = phasemask_diam / (loD * focal_length)

    magnifier = hcipy.Magnifier(lab_diam / telescope_diameter)
    magnified_pupil_grid = hcipy.make_pupil_grid(n_pix_pupil, 2 * lab_diam)

    zwfs = hcipy.ZernikeWavefrontSensorOptics(
        magnified_pupil_grid,
        phase_step=phasemask_depth * np.pi,
        phase_dot_diameter=phase_mask_diam_loD,
        num_pix=128,
        pupil_diameter=lab_diam,
        reference_wavelength=wavelength_wfs,
    )

    wf = hcipy.Wavefront(pupil, wavelength_wfs)
    wf.electric_field /= np.sqrt(wf.intensity.shaped.sum())
    wf = magnifier.forward(wf)
    wf = zwfs.forward(wf)

    if include_cold_stop:
        mag_between_mask_and_stop = 40 / 187
        cold_stop_diameter = 2.15e-3 * mag_between_mask_and_stop

        # print(f"Cold stop diameter: {cold_stop_diameter:.3e} m")

        focal_grid = hcipy.make_focal_grid(
            q=4,
            num_airy=10,
            pupil_diameter=lab_diam,
            reference_wavelength=wavelength_wfs,
            focal_length=254e-3,
        )

        mask = hcipy.make_circular_aperture(cold_stop_diameter)(focal_grid)

        # plt.figure()
        # plt.imshow(mask.shaped)
        cold_stop_ideal = hcipy.OccultedLyotCoronagraph(
            magnified_pupil_grid,
            mask,
            focal_plane_mask_grid=focal_grid,
            focal_length=254e-3,
        )
        wf = cold_stop_ideal.forward(wf)

    img = wf.intensity
    img = hcipy.subsample_field(img, n_pix_pupil / n_pix_final, statistic="sum")
    img = img.shaped

    # plt.figure()
    # plt.imshow(img)
    # plt.colorbar()
    # print(img.sum())

    return np.array(img / img.max())


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
        loss = model_loss
        print(f"Generating model image for beam {beam}, centre {pupil_center}...")
        model_img = generate_zwfs_model_image(
            "AT",
            "H3",
            centre=pupil_center,
            include_cold_stop=True,
        )
        loss_args = (model_img, pupil_mask, 0.1)

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
            loss,
            init_coeffs,
            loss_args,
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
