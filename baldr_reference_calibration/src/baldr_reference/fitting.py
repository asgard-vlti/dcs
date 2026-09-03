#!/usr/bin/env python3
"""Fit Baldr knife-edge and cold-stop alignment from two reference images."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import least_squares, minimize

from .model import generate_references, load_config


PARAMETER_NAMES = (
    "edge_offset_mm",
    "edge_angle_deg",
    "cold_stop_x_um",
    "cold_stop_y_um",
)
DEFAULT_BOUNDS = np.array([
    [-1.35, -0.35],
    [-5.0, 5.0],
    [-180.0, 180.0],
    [-180.0, 180.0],
])
DEFAULT_STEPS = np.array([0.040, 0.25, 40.0, 40.0])


def read_reference_fits(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read clear and masked images without requiring one rigid FITS layout."""
    with fits.open(path) as hdus:
        images = []
        named = {}
        for hdu in hdus:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data, dtype=float)
            if data.ndim == 3 and data.shape[0] >= 2:
                images.extend([data[0], data[1]])
                continue
            if data.ndim != 2:
                continue
            images.append(data)
            name = str(hdu.header.get("EXTNAME", hdu.name)).strip().upper()
            named[name] = data

    clear_aliases = ("CLEAR_PUPIL", "CLEAR", "N0", "MASK_OUT")
    masked_aliases = ("PHASE_MASK", "ZWFS", "I0", "MASK_IN", "REFERENCE")
    clear = next((named[name] for name in clear_aliases if name in named), None)
    masked = next((named[name] for name in masked_aliases if name in named), None)
    if clear is None or masked is None:
        if len(images) < 2:
            raise ValueError("FITS input must contain at least two 2-D images or a two-frame cube.")
        clear, masked = images[:2]
    if clear.shape != masked.shape:
        raise ValueError(f"Clear and ZWFS images must have matching shapes, got {clear.shape} and {masked.shape}.")
    if not np.all(np.isfinite(clear)) or not np.all(np.isfinite(masked)):
        raise ValueError("Reference images contain non-finite values.")
    return clear, masked


def measure_pupil(image: np.ndarray) -> dict:
    """Estimate background, support, centre, and footprint from a clear image."""
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    sigma = 1.4826 * mad
    signal = np.clip(image - background, 0, None)
    smooth = ndimage.gaussian_filter(signal, 1.0)
    peak = float(smooth.max())
    threshold = max(5.0 * sigma, 0.05 * peak)
    labels, count = ndimage.label(smooth > threshold)
    if count == 0:
        raise RuntimeError("Could not identify an illuminated pupil in the clear image.")
    sizes = ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))
    support = labels == int(np.argmax(sizes) + 1)
    rows, columns = np.nonzero(support)
    # Use the outer support, rather than the intensity centroid, for detector
    # registration. Knife-edge and cold-stop misalignment can move the
    # intensity centroid; treating that motion as registration would remove
    # part of the signal that this routine is intended to fit.
    center_y = 0.5 * (rows.min() + rows.max())
    center_x = 0.5 * (columns.min() + columns.max())
    height = int(rows.max() - rows.min() + 1)
    width = int(columns.max() - columns.min() + 1)
    return {
        "background": background,
        "noise_sigma": sigma,
        "threshold": threshold,
        "center_x": float(center_x),
        "center_y": float(center_y),
        "width_pixels": width,
        "height_pixels": height,
        "diameter_pixels": float(max(width, height)),
        "support": support,
    }


def parameters_from_config(config: dict) -> np.ndarray:
    relay = config["relay"]
    return np.array([
        float(relay["edge_offset_m"]) * 1e3,
        np.rad2deg(float(relay["edge_angle_rad"])),
        float(relay["cold_stop_x_offset_m"]) * 1e6,
        float(relay["cold_stop_y_offset_m"]) * 1e6,
    ])


def config_with_parameters(config: dict, parameters: np.ndarray) -> dict:
    result = copy.deepcopy(config)
    relay = result["relay"]
    relay["edge_offset_m"] = float(parameters[0]) * 1e-3
    relay["edge_angle_rad"] = np.deg2rad(float(parameters[1]))
    relay["cold_stop_x_offset_m"] = float(parameters[2]) * 1e-6
    relay["cold_stop_y_offset_m"] = float(parameters[3]) * 1e-6
    return result


def _registered_extract(
    image: np.ndarray,
    shape: tuple[int, int],
    model_center_yx: tuple[float, float],
    measured_center_yx: tuple[float, float],
) -> np.ndarray:
    """Sample a model subframe with its pupil at the measured pupil location."""
    rows, columns = np.indices(shape, dtype=float)
    source_rows = rows + model_center_yx[0] - measured_center_yx[0]
    source_columns = columns + model_center_yx[1] - measured_center_yx[1]
    return ndimage.map_coordinates(
        image,
        (source_rows, source_columns),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _refine_registration(
    model_clear: np.ndarray,
    measured_clear: np.ndarray,
    initial_origin_yx: tuple[float, float],
    support: np.ndarray,
) -> tuple[float, float]:
    """Refine a geometric-centre estimate by subpixel clear-pupil matching."""
    fit_region = ndimage.binary_dilation(support, iterations=2)
    data = measured_clear[fit_region]

    def cost(origin):
        patch = _registered_extract(
            model_clear, measured_clear.shape, tuple(origin), (0.0, 0.0)
        )
        design = np.column_stack((patch[fit_region], np.ones(data.size)))
        scale, background = np.linalg.lstsq(design, data, rcond=None)[0]
        residual = scale * patch[fit_region] + background - data
        return float(np.mean(residual**2))

    initial = np.asarray(initial_origin_yx, dtype=float)
    result = minimize(
        cost,
        initial,
        method="Powell",
        bounds=[(value - 2.0, value + 2.0) for value in initial],
        options={"xtol": 1e-3, "ftol": 1e-8, "maxiter": 80},
    )
    return tuple(np.asarray(result.x, dtype=float))


def solve_photometry(
    model_clear: np.ndarray,
    model_masked: np.ndarray,
    measured_clear: np.ndarray,
    measured_masked: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Solve one shared flux scale and one background per image."""
    mc = model_clear[fit_mask].ravel()
    mz = model_masked[fit_mask].ravel()
    dc = measured_clear[fit_mask].ravel()
    dz = measured_masked[fit_mask].ravel()
    design = np.zeros((mc.size + mz.size, 3))
    design[:mc.size, 0] = mc
    design[mc.size:, 0] = mz
    design[:mc.size, 1] = 1.0
    design[mc.size:, 2] = 1.0
    scale, clear_background, masked_background = np.linalg.lstsq(
        design, np.concatenate((dc, dz)), rcond=None
    )[0]
    return max(float(scale), 1e-12), float(clear_background), float(masked_background)


class AlignmentObjective:
    def __init__(self, measured_clear, measured_masked, config, material_path):
        self.measured_clear = measured_clear
        self.measured_masked = measured_masked
        self.config = config
        self.material_path = material_path
        self.measurement = measure_pupil(measured_clear)
        registration_config = config.get("fit", {}).get("registration", "auto")
        if registration_config not in ("auto", "frame_center"):
            raise ValueError("fit.registration must be 'auto' or 'frame_center'.")

        # Determine registration once from the nominal clear-pupil model and
        # keep it fixed while fitting physical alignment parameters.
        nominal_clear, _, _ = generate_references(config, material_path)
        if nominal_clear.shape[0] < measured_clear.shape[0] or nominal_clear.shape[1] < measured_clear.shape[1]:
            raise ValueError(
                "The theoretical detector image must be at least as large as the measured subframe. "
                "Set detector.crop to null and/or increase the theoretical array size."
            )
        nominal_measurement = measure_pupil(nominal_clear)
        if registration_config == "auto":
            self.model_center_yx = (
                nominal_measurement["center_y"], nominal_measurement["center_x"]
            )
            self.measured_center_yx = (
                self.measurement["center_y"], self.measurement["center_x"]
            )
        else:
            self.model_center_yx = tuple((np.asarray(nominal_clear.shape) - 1.0) / 2.0)
            self.measured_center_yx = tuple((np.asarray(measured_clear.shape) - 1.0) / 2.0)
        self.registration = registration_config
        initial_crop_origin_yx = (
            self.model_center_yx[0] - self.measured_center_yx[0],
            self.model_center_yx[1] - self.measured_center_yx[1],
        )
        if registration_config == "auto" and nominal_clear.shape != measured_clear.shape:
            self.crop_origin_yx = _refine_registration(
                nominal_clear,
                measured_clear,
                initial_crop_origin_yx,
                self.measurement["support"],
            )
        else:
            self.crop_origin_yx = initial_crop_origin_yx
        measured_support = self.measurement["support"] | measure_pupil(measured_masked)["support"]
        self.fit_mask = ndimage.binary_dilation(measured_support, iterations=2)
        reference_level = np.median(
            np.concatenate((measured_clear[self.fit_mask], measured_masked[self.fit_mask]))
        )
        self.sigma = max(abs(float(reference_level)) * 2e-3, 1e-12)
        self.cache = {}

    def model(self, parameters):
        key = tuple(np.asarray(parameters, dtype=float))
        if key not in self.cache:
            trial = config_with_parameters(self.config, np.asarray(parameters))
            clear, masked, _ = generate_references(trial, self.material_path)
            clear = _registered_extract(
                clear, self.measured_clear.shape, self.crop_origin_yx, (0.0, 0.0)
            )
            masked = _registered_extract(
                masked, self.measured_masked.shape, self.crop_origin_yx, (0.0, 0.0)
            )
            photometry = solve_photometry(
                clear, masked, self.measured_clear, self.measured_masked, self.fit_mask
            )
            scale, clear_background, masked_background = photometry
            clear = scale * clear + clear_background
            masked = scale * masked + masked_background
            self.cache[key] = clear, masked, photometry
        return self.cache[key]

    def residual(self, parameters):
        clear, masked, _ = self.model(parameters)
        return np.concatenate((
            (clear[self.fit_mask] - self.measured_clear[self.fit_mask]) / self.sigma,
            (masked[self.fit_mask] - self.measured_masked[self.fit_mask]) / self.sigma,
        ))

    def cost(self, parameters):
        residual = self.residual(parameters)
        return float(np.mean(residual**2))


def coarse_pair(objective, parameters, bounds, pair, points):
    x_values = np.linspace(bounds[pair[0], 0], bounds[pair[0], 1], points)
    y_values = np.linspace(bounds[pair[1], 0], bounds[pair[1], 1], points)
    costs = np.empty((points, points))
    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            trial = parameters.copy()
            trial[pair[0]] = x_value
            trial[pair[1]] = y_value
            costs[row, column] = objective.cost(trial)
    best = np.unravel_index(np.argmin(costs), costs.shape)
    result = parameters.copy()
    result[pair[0]] = x_values[best[1]]
    result[pair[1]] = y_values[best[0]]
    return result, x_values, y_values, costs


def finite_difference_jacobian(objective, parameters, bounds, steps):
    at_parameters = objective.residual(parameters)
    jacobian = np.empty((at_parameters.size, len(parameters)))
    for index, step in enumerate(steps):
        trial = parameters.copy()
        direction = step if parameters[index] + step <= bounds[index, 1] else -step
        trial[index] += direction
        jacobian[:, index] = (objective.residual(trial) - at_parameters) / direction
    return jacobian


def fit_alignment(
    measured_clear, measured_masked, config, output_directory,
    write_diagnostics=True,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    material_path = Path(__file__).with_name("Exposed_Ma-N_1405_optical_constants.txt")
    settings = config.get("fit", {})
    bounds = np.asarray(settings.get("bounds", DEFAULT_BOUNDS.tolist()), dtype=float)
    steps = np.asarray(settings.get("finite_difference_steps", DEFAULT_STEPS.tolist()), dtype=float)
    points = int(settings.get("coarse_points_per_axis", 5))
    passes = int(settings.get("coarse_passes", 1))
    objective = AlignmentObjective(measured_clear, measured_masked, config, material_path)
    parameters = np.clip(parameters_from_config(config), bounds[:, 0], bounds[:, 1])
    maps = []
    for _ in range(passes):
        for pair in ((0, 1), (2, 3)):
            parameters, x_values, y_values, costs = coarse_pair(
                objective, parameters, bounds, pair, points
            )
            maps.append((pair, x_values, y_values, costs))

    result = least_squares(
        objective.residual,
        parameters,
        jac=lambda p: finite_difference_jacobian(objective, p, bounds, steps),
        bounds=(bounds[:, 0], bounds[:, 1]),
        x_scale=np.maximum(steps, 1e-12),
        max_nfev=int(settings.get("max_function_evaluations", 30)),
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )
    fitted_clear, fitted_masked, photometry = objective.model(result.x)

    if write_diagnostics:
        fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 5), squeeze=False)
        for axis, (pair, x_values, y_values, costs) in zip(axes[0], maps):
            image = axis.imshow(
                np.log10(costs), origin="lower", aspect="auto",
                extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
            )
            axis.set_xlabel(PARAMETER_NAMES[pair[0]])
            axis.set_ylabel(PARAMETER_NAMES[pair[1]])
            axis.set_title("Coarse log10 mean squared residual")
            fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(output_directory / "coarse_fit_maps.png", dpi=180)
        plt.close(fig)

    summary = {
        "parameter_names": list(PARAMETER_NAMES),
        "initial_parameters": parameters_from_config(config).tolist(),
        "coarse_parameters": parameters.tolist(),
        "fitted_parameters": result.x.tolist(),
        "bounds": bounds.tolist(),
        "flux_scale": photometry[0],
        "clear_background": photometry[1],
        "zwfs_background": photometry[2],
        "pupil_measurement": {k: v for k, v in objective.measurement.items() if k != "support"},
        "registration": objective.registration,
        "model_pupil_center_yx": list(objective.model_center_yx),
        "measured_pupil_center_yx": list(objective.measured_center_yx),
        "model_crop_origin_yx": list(objective.crop_origin_yx),
        "fit_success": bool(result.success),
        "fit_message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "nfev": int(result.nfev),
        "config": config,
    }
    (output_directory / "fit_summary.json").write_text(json.dumps(summary, indent=2))
    if write_diagnostics:
        primary = fits.PrimaryHDU(measured_clear)
        primary.header["EXTNAME"] = "MEASURED_CLEAR"
        fits.HDUList([
            primary,
            fits.ImageHDU(measured_masked, name="MEASURED_ZWFS"),
            fits.ImageHDU(fitted_clear, name="MODEL_CLEAR"),
            fits.ImageHDU(fitted_masked, name="MODEL_ZWFS"),
            fits.ImageHDU(measured_clear - fitted_clear, name="RESID_CLEAR"),
            fits.ImageHDU(measured_masked - fitted_masked, name="RESID_ZWFS"),
            fits.ImageHDU(objective.fit_mask.astype(np.uint8), name="FIT_MASK"),
        ]).writeto(output_directory / "fit_products.fits", overwrite=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=Path, help="FITS containing clear and ZWFS images")
    parser.add_argument("config", type=Path, help="Standalone model and fit configuration")
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    clear, masked = read_reference_fits(args.references)
    summary = fit_alignment(clear, masked, load_config(args.config), args.output_directory)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
