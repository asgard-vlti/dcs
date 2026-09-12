import numpy as np
import numpy as onp
import zmq
import time
import argparse
from pathlib import Path
import subprocess

from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import hcipy
import scipy.optimize as opt
import scipy.ndimage as ndi

# Current calibration products are maintained externally at the fixed beam paths.
USE_FITS = True


def create_parser():
    parser = argparse.ArgumentParser(
        description="Flatten beam wavefront using hardware in the loop optimization."
    )
    parser.add_argument("beam", type=int, help="Beam number")

    parser.add_argument(
        "--target",
        choices=["stddev", "model", "amp-model"],
        default="model",
        help="Whether the target should be visual flatness or a model based reference. "
        "The model is generated using the pupil only image and the propagation system. "
        "With saved-ref, model and amp-model both use the saved ZWFS image directly.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Suppress plots at the end of optimization",
    )
    parser.add_argument(
        "--pupil",
        type=str,
        choices=["Lab", "AT", "UT"],
        default="Lab",
        help="Which pupil to use for the model image generation. Only used if target is 'model' or 'amp-model'.",
    )
    parser.add_argument(
        "--mask",
        choices=[*(f"J{i}" for i in range(1, 6)), *(f"H{i}" for i in range(1, 6))],
        default="H3",
        help="Which ZWFS mask to use for the model image generation. Only used if target is 'model' or 'amp-model'.",
    )

    parser.add_argument(
        "--source",
        choices=["live", "saved-pupil", "saved-ref"],
        default="live",
        help="Use a live clear pupil (default), a saved clear pupil from "
        "~/etc/b-pupils/beam{beam}.fits, or a saved ZWFS reference from "
        "~/etc/b-references/beam{beam}.fits. FITS images use CLEAR_PUPIL or "
        "PHASE_MASK respectively. Set USE_FITS=False in the script to load .npy "
        "instead. All sources acquire a fresh dark and optimize against live frames.",
    )
    return parser


def parse_args(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.source == "saved-ref" and args.target == "stddev":
        parser.error("--source saved-ref requires --target model or amp-model")
    return args


def load_saved_image(beam, source):
    """Load a calibrated detector image; never subtract the current live dark."""
    directory, extension = {
        "saved-pupil": ("b-pupils", "CLEAR_PUPIL"),
        "saved-ref": ("b-references", "PHASE_MASK"),
    }[source]
    suffix = ".fits" if USE_FITS else ".npy"
    path = Path.home() / "etc" / directory / f"beam{beam}{suffix}"
    try:
        if USE_FITS:
            with fits.open(path) as hdus:
                data = onp.array(hdus[extension].data, copy=True)
        else:
            data = onp.load(path, allow_pickle=False)
        if data.dtype.kind not in "fiu" or data.shape != (32, 32):
            raise ValueError("expected a numeric 32x32 image")
        image = onp.array(data, dtype=float, copy=True)
        flux = image.sum()
        if not onp.isfinite(image).all() or not onp.isfinite(flux) or flux <= 0:
            raise ValueError("expected finite pixels and positive finite total flux")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot load {source} from {path}: {exc}") from exc
    print(f"Loaded {source} from {path}")
    return image


TARGET_TO_FLAT_NAME = {
    "stddev": "night-standard",
    "model": "test",
}


def amp_aberrated_aperture(amp_errors, center, secondary_ratio, subsample=True):
    n_pix_pupil = 128
    n_pix_final = 32
    telescope_diameter = 12e-3
    secondary_diameter = secondary_ratio * telescope_diameter
    aperture = hcipy.make_obstructed_circular_aperture(
        pupil_diameter=telescope_diameter,
        central_obscuration_ratio=secondary_diameter / telescope_diameter,
        num_spiders=0,
    )

    pupil_grid = hcipy.make_pupil_grid(n_pix_pupil, 2 * telescope_diameter)
    pupil_grid = pupil_grid.shift(-center * telescope_diameter / n_pix_final)

    amp_errors /= np.mean(amp_errors)

    pupil = hcipy.evaluate_supersampled(aperture, pupil_grid, 6)

    pupil *= ndi.gaussian_filter(amp_errors, sigma=2.0).flatten()

    if subsample:
        pupil = hcipy.subsample_field(pupil, n_pix_pupil / n_pix_final, statistic="sum")

    return np.array(pupil.shaped)


def loss(params, args):
    (img,) = args
    centre = np.array((params[1], params[2]))
    secondary_ratio = params[3]
    amp_errors = params[4:].reshape((128, 128))
    model = amp_aberrated_aperture(amp_errors, centre, secondary_ratio)

    model /= np.sum(model)
    img /= np.sum(img)
    return -np.sum(img * model) * 1e3


def fit_amp_errors(pupil_img, pupil_radius, pupil_center, secondary_ratio, out_scale=4):
    downscale = out_scale
    amp_errors = np.kron(pupil_img, np.ones((downscale, downscale)))

    init_params = np.concatenate(
        (
            np.array(
                [
                    pupil_radius,
                    pupil_center[0],
                    pupil_center[1],
                    secondary_ratio,
                ]
            ),
            (amp_errors / amp_errors.mean()).flatten(),
        )
    )

    res = opt.minimize(
        loss,
        x0=init_params,
        args=((pupil_img,),),
        bounds=((8, 8), (-10, 10), (-10, 10), (0.05, 0.2))
        + ((0.0, 10.5),) * (128 * 128),
        # options={"maxiter": 100},
        method="L-BFGS-B",
    )

    return res


def get_telescope_params(case):
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

    return telescope_diameter, secondary_diameter, aperture, lab_diam


class AmpAberratedAperture:
    def __init__(self, basis, radius, center, secondary_ratio):
        self.basis = basis
        self.radius = radius
        self.center = center
        self.secondary_ratio = secondary_ratio

    def get_pupil_noerr(self):

        # build the aperture as a circular obstruction with the given radius and center
        pupil_grid = hcipy.make_pupil_grid(256, 2)
        telescope_diameter = self.radius * 2 / 16
        secondary_diameter = self.secondary_ratio * telescope_diameter
        aperture = hcipy.make_obstructed_circular_aperture(
            pupil_diameter=telescope_diameter,
            central_obscuration_ratio=secondary_diameter / telescope_diameter,
            num_spiders=0,
        )
        pupil_grid = pupil_grid.shift(-self.center * telescope_diameter / 32)

        pupil_noerr = hcipy.evaluate_supersampled(aperture, pupil_grid, 6)

        return pupil_noerr

    def get_image(self, amp_coeffs, subsample=True):
        """
        amp_coeffs are now the coefficients for the basis
        """
        pupil_noerr = self.get_pupil_noerr()

        amp_errors = self.basis.linear_combination(amp_coeffs)

        amp = pupil_noerr * (1 + amp_errors)

        if subsample:
            amp = hcipy.subsample_field(amp, 256 / 32, statistic="sum")

        return np.array(amp.shaped) / np.sum(np.array(amp.shaped))


# %%
def aaa_loss(params, args):
    radius, center_x, center_y, secondary_ratio = params[:4]
    amp_coeffs = params[4:]
    img, basis = args
    aaa = AmpAberratedAperture(
        basis, radius, np.array([center_x, center_y]), secondary_ratio
    )
    model = aaa.get_image(amp_coeffs)
    model /= np.sum(model)

    img = img / np.sum(img)
    rmse = np.sqrt(np.mean((model - img) ** 2))
    return rmse


_epsilon = 1e-12


def make_fourier_basis(grid, fourier_grid, sort_by_energy=True):
    """
    Taken from HCIPY, adapted to return the frequencies too

    Make a Fourier basis.

    Fourier modes this function are defined to be real. This means that for each point, both a sine and cosine mode is returned.

    Repeated frequencies will not be repeated in this mode basis. This means that opposite points in the `fourier_grid` will be silently ignored.

    Parameters
    ----------
    grid : Grid
        The :class:`Grid` on which to calculate the modes.
    fourier_grid : Grid
        The grid defining all frequencies.
    sort_by_energy : bool
        Whether to sort by increasing energy or not.

    Returns
    -------
    ModeBasis
        The mode basis containing all Fourier modes.
    """
    modes_cos = []
    modes_sin = []
    energies = []
    ignore_list = []
    freqs = []

    c = np.array(grid.coords)

    for i, p in enumerate(fourier_grid.points):
        if i in ignore_list:
            continue

        mode_cos = hcipy.Field(np.cos(np.dot(p, c)), grid)
        mode_sin = hcipy.Field(np.sin(np.dot(p, c)), grid)

        modes_cos.append(mode_cos)
        modes_sin.append(mode_sin)

        j = fourier_grid.closest_to(-p)

        dist = fourier_grid.points[j] + p
        dist2 = np.dot(dist, dist)

        p_length2 = np.dot(p, p)
        energies.append(p_length2)
        freqs.append(p)

        if dist2 < (_epsilon * p_length2):
            ignore_list.append(j)

    if sort_by_energy:
        ind = np.argsort(energies)
        modes_sin = [modes_sin[i] for i in ind]
        modes_cos = [modes_cos[i] for i in ind]
        freqs = [freqs[i] for i in ind]
        energies = np.array(energies)[ind]

    modes = []
    mode_freqs = []
    for i, E in enumerate(energies):
        # Filter out and correctly normalize zero energy vs non-zero energy modes.
        if E > _epsilon:
            modes.append(modes_cos[i] * np.sqrt(2))
            modes.append(modes_sin[i] * np.sqrt(2))
            mode_freqs.append(freqs[i])
            mode_freqs.append(freqs[i])
        else:
            modes.append(modes_cos[i])
            mode_freqs.append(freqs[i])

    return hcipy.ModeBasis(modes, grid), np.array(mode_freqs)


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
    centre=None,
    pupil_guess=None,
    include_cold_stop=True,
    n_pix_pupil=256,
    n_pix_final=32,
):
    """
    Assumes pupil_guess is in intensity, not amplitude!!
    """

    # validate all inputs
    if case not in ["AT", "UT", "Lab"]:
        raise ValueError(f"Invalid case: {case}. Must be one of ['AT', 'UT', 'Lab']")
    if phasemask not in phasemask_parameters.keys():
        raise ValueError(
            f"Invalid phasemask: {phasemask}. Must be one of {phasemask_parameters.keys()}"
        )
    if centre is None and pupil_guess is None:
        raise ValueError("Must provide either centre or pupil_guess")
    if centre is not None and pupil_guess is not None:
        raise ValueError("Must provide either centre or pupil_guess, not both")

    if phasemask.startswith("J"):
        wavelength_wfs = 1.25e-6
    elif phasemask.startswith("H"):
        wavelength_wfs = 1.65e-6

    phasemask_diam = phasemask_parameters[phasemask]["diameter"] * 1e-6
    phasemask_depth = phasemask_parameters[phasemask]["depth"]

    telescope_diameter, secondary_diameter, aperture, lab_diam = get_telescope_params(
        case
    )

    # convert centre from pixels to physical units
    if centre is not None:
        centre = centre.copy()
        centre -= np.array([(n_pix_final - 1) / 2, (n_pix_final - 1) / 2])
        centre = 2 * centre * telescope_diameter / n_pix_final

        pupil_grid = hcipy.make_pupil_grid(n_pix_pupil, 2 * telescope_diameter)
        pupil_grid = pupil_grid.shift(-centre)
        pupil = hcipy.evaluate_supersampled(aperture, pupil_grid, 6)

    if pupil_guess is not None:
        pupil = pupil_guess.copy()
        pupil /= np.max(pupil)

        downscale = n_pix_pupil // pupil.shape[0]
        pupil = np.kron(pupil, np.ones((downscale, downscale)))

        pupil = hcipy.Field(
            np.sqrt(pupil.flatten()),
            hcipy.make_pupil_grid(n_pix_pupil, 2 * telescope_diameter),
        )

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


def acquire_pupil(beam, cam, sock, mds_send):
    print(f"Taking pupil only image for beam {beam}...")
    offset = 200.0
    mds_send(sock, f"moverel BMX{beam} {offset}")
    mds_send(sock, f"moverel BMY{beam} {offset}")
    time.sleep(1)

    pupil_only = cam.take_stack(1000).mean(0)

    mds_send(sock, f"moverel BMX{beam} {-offset}")
    mds_send(sock, f"moverel BMY{beam} {-offset}")
    time.sleep(1)

    # if show_plots:
    #     plt.imshow(pupil_only)
    #     plt.colorbar()
    #     plt.show()

    return pupil_only


def prepare_pupil(pupil_only, show_plots=False):
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
    pupil_radius = res.x[0]

    # if show_plots:
    #     plt.imshow(pupil_only)
    #     plt.contour(pupil_mask, levels=[0.5], colors="r")
    #     plt.show()

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

    scattered_flux_mask /= scattered_flux_mask.sum()
    return pupil_mask, scattered_flux_mask, pupil_center, pupil_radius


def generate_target_image(
    target, beam, pupil_only, pupil_center, pupil_radius, show_plots=False
):
    if target == "model":
        print(f"Generating model image for beam {beam}, centre {pupil_center}...")
        model_img = generate_zwfs_model_image(
            "Lab",
            "H3",
            centre=np.array(pupil_center) + (32 - 1) / 2,
            include_cold_stop=True,
        )
    elif target == "amp-model":
        print("fitting amplitude errors to pupil only image...")
        pupil_grid = hcipy.make_pupil_grid(256, 2)

        freqs = []
        start_HO = 0.0
        max_freq_HO = 4.0
        min_freq_HO = 0.5
        spacing_HO = 1.0
        n_accross = int((max_freq_HO - start_HO) / spacing_HO) + 1
        u = start_HO
        for i in range(n_accross):
            v = start_HO
            for j in range(n_accross):
                if (
                    np.sqrt(u**2 + v**2) <= max_freq_HO
                    and np.sqrt(u**2 + v**2) >= min_freq_HO
                ):
                    freqs.append([u, v])
                    if np.abs(u) > 1e-6 and v > 1e-6:
                        freqs.append([-u, v])
                v += spacing_HO
            u += spacing_HO
        freqs = np.array(freqs)

        basis, freqs = make_fourier_basis(pupil_grid, hcipy.Grid(freqs.T * 2 * np.pi))

        telescope_diameter, secondary_diameter, aperture, lab_diam = (
            get_telescope_params("Lab")
        )

        init_params = np.concatenate(
            (
                np.array(
                    [
                        pupil_radius,
                        pupil_center[0],
                        pupil_center[1],
                        secondary_diameter / telescope_diameter,
                    ]
                ),
                np.zeros(basis.num_modes),
            )
        )

        res = opt.minimize(
            aaa_loss,
            x0=init_params,
            args=((pupil_only / np.sum(pupil_only), basis),),
            bounds=((7, 9), (-10, 10), (-10, 10), (0.05, 0.2))
            + ((-0.1, 0.1),) * basis.num_modes,
            # options={"maxiter": 100},
            method="L-BFGS-B",
            options={
                "eps": 1e-3,
            },
        )

        final_pupil = AmpAberratedAperture(
            basis,
            radius=res.x[0],
            center=np.array([res.x[1], res.x[2]]),
            secondary_ratio=res.x[3],
        ).get_image(res.x[4:], subsample=False)

        print(f"Generating model image for beam {beam}, using pupil image...")

        model_img = generate_zwfs_model_image(
            case="Lab",
            phasemask="H3",
            pupil_guess=final_pupil,
            include_cold_stop=True,
            n_pix_pupil=256,
            n_pix_final=32,
        )

        if show_plots:
            plt.figure()
            plt.subplot(121)
            plt.imshow(pupil_only)
            plt.title("Pupil only image")
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(final_pupil)
            plt.title("Fitted intensity errors")
            plt.colorbar()
            plt.show()

    else:
        raise ValueError(
            f"Invalid model target: {target}. Must be 'model' or 'amp-model'"
        )

    return model_img


def main(argv=None):
    args = parse_args(argv)
    saved_image = None
    if args.source != "live":
        try:
            saved_image = load_saved_image(args.beam, args.source)
        except ValueError as exc:
            create_parser().error(str(exc))

    from asgard_alignment import DM_modes2
    from asgard_alignment.bcam import Bcam
    from asgard_alignment.DM_shm_ctrl import dmclass

    beam = args.beam
    show_plots = not args.no_plots

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
    mds_send(sock, f"moveabs BMY{beam} 0.0")
    time.sleep(3)
    cam.take_dark(256)
    # if show_plots:
    #     plt.imshow(cam.dark)
    #     plt.colorbar()
    #     plt.show()

    mds_send(sock, f"moveabs BMY{beam} {cur_bmy}")
    time.sleep(3)

    if args.source == "live":
        pupil_only = acquire_pupil(beam, cam, sock, mds_send)
    elif args.source == "saved-pupil":
        pupil_only = saved_image

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

    pupil_mask = None
    if args.source != "saved-ref":
        pupil_mask, scattered_flux_mask, pupil_center, pupil_radius = prepare_pupil(
            pupil_only, show_plots
        )

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

        # img_in_pupil = img * pupil_mask
        # model_in_pupil = model_img * pupil_mask
        img_in_pupil = img
        model_in_pupil = model_img

        img_in_pupil /= np.sum(img_in_pupil)
        model_in_pupil /= np.sum(model_in_pupil)

        # return -np.sum(img_in_pupil * model_in_pupil)
        rmse = np.sqrt(np.mean((img_in_pupil - model_in_pupil) ** 2))
        return rmse

    if args.target == "stddev":
        loss = basis_loss
        loss_args = (0.3, scattered_flux_mask, pupil_mask, 0.1)
    else:
        loss = model_loss
        if args.source == "saved-ref":
            model_img = saved_image
        else:
            model_img = generate_target_image(
                args.target, beam, pupil_only, pupil_center, pupil_radius, show_plots
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
            (fourier, *loss_args),
            method="COBYLA",
            options={"disp": True, "maxiter": n_it},
            # bounds=[[-0.05, 0.05] for _ in range(n_modes)],
        )

        print(f"Loss at end of optimization with {n_modes} modes: {res.fun:.3f}")

        prev_fourier = fourier

        if show_plots and args.target in ("model", "amp-model"):
            dm.set_data(fourier.linear_combination(res.x * 0.05))
            time.sleep(0.5)
            img = cam.take_stack(64).mean(0)
            plt.figure()
            plt.subplot(131)
            plt.imshow(model_img / np.sum(model_img))
            plt.colorbar()
            plt.title("Model image")
            plt.subplot(132)
            plt.imshow(img / np.sum(img))
            plt.colorbar()
            plt.title("Current image")
            plt.subplot(133)
            plt.imshow(
                model_img / np.sum(model_img) - img / np.sum(img),
                norm=mcolors.CenteredNorm(),
                cmap="RdBu_r",
            )
            plt.colorbar()
            plt.title("Difference")
            plt.show()

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
