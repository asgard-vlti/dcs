#!/usr/bin/env python3
"""Fit an on-sky pupil amplitude and generate updated Baldr references."""

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
from scipy.optimize import least_squares

from .fitting import _registered_extract, measure_pupil
from .generate import add_complete_config
from .model import generate_clear_reference, generate_references, load_config, make_pupil


PARAMETER_NAMES = (
    "rotation_deg",
    "shift_x_pupil_pixels",
    "shift_y_pupil_pixels",
    "diameter_scale",
)


def estimate_pupil_center_subpixel(image: np.ndarray) -> dict:
    """Fit the outer pupil edge with a robust circle at subpixel precision.

    The fit uses only strong gradients in an annulus around the outer pupil
    boundary.  This suppresses the central obstruction, spiders, and smooth
    illumination structure, none of which should define the pupil centre.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError("The pupil-centre image must be a finite 2-D array.")

    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    noise_sigma = max(1.4826 * mad, np.finfo(float).eps)
    signal = np.clip(image - background, 0.0, None)
    smooth = ndimage.gaussian_filter(signal, 0.55)
    peak = float(smooth.max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("The clear pupil contains no positive signal.")

    threshold = max(5.0 * noise_sigma, 0.08 * peak)
    labels, count = ndimage.label(smooth > threshold)
    if count == 0:
        raise RuntimeError("Could not identify the clear-pupil support.")
    sizes = np.asarray(
        ndimage.sum(np.ones_like(labels), labels, range(1, count + 1)),
        dtype=float,
    )
    largest = float(sizes.max())
    retained = np.flatnonzero(sizes >= max(4.0, 0.08 * largest)) + 1
    support = np.isin(labels, retained)
    rows, columns = np.nonzero(support)
    if rows.size < 12:
        raise RuntimeError("Too few illuminated pixels for a pupil-centre fit.")

    center_y0 = 0.5 * (rows.min() + rows.max())
    center_x0 = 0.5 * (columns.min() + columns.max())
    radius0 = 0.25 * (
        rows.max() - rows.min() + columns.max() - columns.min() + 2.0
    )
    yy, xx = np.indices(image.shape, dtype=float)
    radius = np.hypot(xx - center_x0, yy - center_y0)

    gradient_y = ndimage.sobel(smooth, axis=0, mode="nearest") / 8.0
    gradient_x = ndimage.sobel(smooth, axis=1, mode="nearest") / 8.0
    gradient = np.hypot(gradient_x, gradient_y)
    annulus_half_width = max(1.5, 0.18 * radius0)
    annulus = np.abs(radius - radius0) <= annulus_half_width
    candidate_gradient = gradient[annulus]
    positive = candidate_gradient[candidate_gradient > 0.0]
    if positive.size < 12:
        raise RuntimeError("The outer pupil edge is too weak for a centre fit.")
    gradient_threshold = max(
        float(np.percentile(positive, 55.0)),
        3.0 * noise_sigma,
    )
    edge = annulus & (gradient >= gradient_threshold)
    edge_y, edge_x = np.nonzero(edge)
    if edge_y.size < 12:
        raise RuntimeError("Too few outer-edge samples for a pupil-centre fit.")

    weights = gradient[edge]
    weights /= max(float(np.median(weights)), np.finfo(float).eps)
    sqrt_weights = np.sqrt(np.clip(weights, 0.1, 10.0))

    def residual(parameters):
        center_x, center_y, fitted_radius = parameters
        distances = np.hypot(edge_x - center_x, edge_y - center_y)
        return sqrt_weights * (distances - fitted_radius)

    maximum_radius = float(np.hypot(*image.shape))
    result = least_squares(
        residual,
        np.array([center_x0, center_y0, radius0]),
        bounds=(
            np.array([0.0, 0.0, 1.0]),
            np.array([image.shape[1] - 1.0, image.shape[0] - 1.0, maximum_radius]),
        ),
        loss="soft_l1",
        f_scale=0.35,
        x_scale=np.array([0.25, 0.25, 0.5]),
        max_nfev=200,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    center_x, center_y, fitted_radius = map(float, result.x)

    # Refine the centre against the complete outer-edge intensity transition.
    # A planar illumination term prevents smooth pupil gradients from pulling
    # the centre, while the robust loss rejects spiders and local defects.
    profile_radius = np.hypot(xx - center_x, yy - center_y)
    profile_region = (
        np.abs(profile_radius - fitted_radius)
        <= max(3.0, 0.40 * fitted_radius)
    )
    profile_x = xx[profile_region]
    profile_y = yy[profile_region]
    profile_data = smooth[profile_region]
    residual_scale = max(noise_sigma, 1e-3 * peak)

    def profile_residual(parameters):
        (
            trial_x, trial_y, trial_radius, edge_width,
            level, slope_x, slope_y, local_background,
        ) = parameters
        distance = np.hypot(profile_x - trial_x, profile_y - trial_y)
        transition = 0.5 * (1.0 - np.tanh(
            (distance - trial_radius) / edge_width
        ))
        illumination = (
            level
            + slope_x * (profile_x - trial_x) / trial_radius
            + slope_y * (profile_y - trial_y) / trial_radius
        )
        return (local_background + illumination * transition - profile_data) / residual_scale

    profile_result = least_squares(
        profile_residual,
        np.array([
            center_x, center_y, fitted_radius, 0.6,
            peak, 0.0, 0.0, 0.0,
        ]),
        bounds=(
            np.array([
                center_x - 2.0, center_y - 2.0, 0.70 * fitted_radius,
                0.15, 0.20 * peak, -peak, -peak, -0.20 * peak,
            ]),
            np.array([
                center_x + 2.0, center_y + 2.0, 1.30 * fitted_radius,
                2.0, 2.0 * peak, peak, peak, 0.20 * peak,
            ]),
        ),
        loss="soft_l1",
        f_scale=2.0,
        x_scale=np.array([
            0.20, 0.20, 0.50, 0.20,
            peak, 0.20 * peak, 0.20 * peak, residual_scale,
        ]),
        max_nfev=500,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    center_x, center_y, fitted_radius = map(float, profile_result.x[:3])

    angles = np.mod(np.arctan2(edge_y - center_y, edge_x - center_x), 2 * np.pi)
    occupied = np.unique(np.floor(angles / (2 * np.pi) * 24).astype(int))
    angular_coverage = float(occupied.size / 24.0)
    if not profile_result.success:
        raise RuntimeError(
            f"Subpixel pupil-centre fit failed: {profile_result.message}"
        )
    if angular_coverage < 0.35:
        raise RuntimeError(
            "Subpixel pupil-centre fit has insufficient outer-edge coverage "
            f"({angular_coverage:.0%})."
        )
    radial_residual = np.hypot(edge_x - center_x, edge_y - center_y) - fitted_radius
    radial_rms = float(np.sqrt(np.mean(radial_residual**2)))

    center_sigma_x = np.nan
    center_sigma_y = np.nan
    if profile_result.jac.shape[0] > profile_result.jac.shape[1]:
        try:
            covariance = np.linalg.inv(profile_result.jac.T @ profile_result.jac)
            residual_variance = float(
                np.sum(profile_result.fun**2)
                / (profile_result.jac.shape[0] - profile_result.jac.shape[1])
            )
            covariance *= residual_variance
            center_sigma_x = float(np.sqrt(max(covariance[0, 0], 0.0)))
            center_sigma_y = float(np.sqrt(max(covariance[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            pass

    return {
        "center_x": center_x,
        "center_y": center_y,
        "radius_pixels": fitted_radius,
        "center_sigma_x_pixels": center_sigma_x,
        "center_sigma_y_pixels": center_sigma_y,
        "radial_residual_rms_pixels": radial_rms,
        "angular_coverage_fraction": angular_coverage,
        "edge_sample_count": int(edge_y.size),
        "method": "robust outer-edge profile with planar illumination",
        "fit_success": bool(profile_result.success),
        "fit_message": str(profile_result.message),
    }


def read_clear_fits(path: str | Path) -> np.ndarray:
    """Read a clear pupil from either a single-image or reference-pair FITS."""
    with fits.open(path) as hdus:
        first_image = None
        for hdu in hdus:
            if hdu.data is None:
                continue
            data = np.asarray(hdu.data, dtype=float)
            if data.ndim == 3 and data.shape[0] >= 1:
                data = data[0]
            if data.ndim != 2:
                continue
            if first_image is None:
                first_image = data
            name = str(hdu.header.get("EXTNAME", hdu.name)).strip().upper()
            if name in ("CLEAR_PUPIL", "CLEAR", "N0", "MASK_OUT"):
                first_image = data
                break
    if first_image is None:
        raise ValueError("FITS input does not contain a 2-D clear-pupil image.")
    if not np.all(np.isfinite(first_image)):
        raise ValueError("The clear-pupil image contains non-finite values.")
    return first_image


def transform_pupil(
    pupil: np.ndarray,
    rotation_deg: float,
    shift_x_pixels: float,
    shift_y_pixels: float,
    diameter_scale: float,
) -> np.ndarray:
    """Rotate, translate, and scale a pupil on its high-resolution grid."""
    if not np.isfinite(diameter_scale) or diameter_scale <= 0:
        raise ValueError("diameter_scale must be finite and positive.")
    angle = np.deg2rad(float(rotation_deg))
    cosine = np.cos(angle) / diameter_scale
    sine = np.sin(angle) / diameter_scale
    matrix = np.array([[cosine, -sine], [sine, cosine]])
    center_yx = (np.asarray(pupil.shape, dtype=float) - 1.0) / 2.0
    shifted_center_yx = center_yx + np.array(
        [float(shift_y_pixels), float(shift_x_pixels)]
    )
    offset = center_yx - matrix @ shifted_center_yx
    transformed = ndimage.affine_transform(
        np.asarray(pupil, dtype=float),
        matrix,
        offset=offset,
        output_shape=pupil.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.clip(transformed, 0.0, 1.0)


def correction_from_grid(coefficients: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly interpolate a coarse multiplicative-amplitude correction."""
    coefficients = np.asarray(coefficients, dtype=float)
    rows = np.linspace(0.0, coefficients.shape[0] - 1.0, shape[0])
    columns = np.linspace(0.0, coefficients.shape[1] - 1.0, shape[1])
    column_grid, row_grid = np.meshgrid(columns, rows)
    return ndimage.map_coordinates(
        coefficients,
        (row_grid, column_grid),
        order=1,
        mode="nearest",
        prefilter=False,
    )


def _difference_matrix(shape: tuple[int, int]) -> np.ndarray:
    rows = []
    count = shape[0] * shape[1]
    for y in range(shape[0]):
        for x in range(shape[1]):
            index = y * shape[1] + x
            if x + 1 < shape[1]:
                row = np.zeros(count)
                row[index] = 1.0
                row[index + 1] = -1.0
                rows.append(row)
            if y + 1 < shape[0]:
                row = np.zeros(count)
                row[index] = 1.0
                row[index + shape[1]] = -1.0
                rows.append(row)
    return np.asarray(rows)


def _solve_photometry(model: np.ndarray, measured: np.ndarray, mask: np.ndarray):
    design = np.column_stack((model[mask], np.ones(np.count_nonzero(mask))))
    scale, background = np.linalg.lstsq(design, measured[mask], rcond=None)[0]
    return max(float(scale), 1e-12), float(background)


class PupilObjective:
    """Clear-pupil objective with fixed relay alignment and detector registration."""

    def __init__(self, measured_clear: np.ndarray, config: dict):
        if measured_clear.ndim != 2 or not np.all(np.isfinite(measured_clear)):
            raise ValueError("The measured clear pupil must be a finite 2-D image.")
        self.measured_clear = np.asarray(measured_clear, dtype=float)
        self.config = copy.deepcopy(config)
        self.config.setdefault("detector", {})["crop"] = None
        self.settings = config.get("pupil_fit", {})

        prior_config = copy.deepcopy(self.config)
        prior_config["pupil"]["rotation_deg"] = 0.0
        self.prior_pupil = make_pupil(prior_config)
        pupil_config = config["pupil"]
        geometry = str(pupil_config.get("geometry") or "").strip().lower()
        spiders = pupil_config.get("spiders") or {}
        self.circular_prior = geometry in {"solarstein", "disc", "disk"} or (
            not geometry and int(spiders.get("count", 0)) == 0
        )

        nominal_parameters = np.array([
            float(config["pupil"].get("rotation_deg", 0.0)),
            0.0,
            0.0,
            1.0,
        ])
        nominal_amplitude = self.amplitude_from_parameters(nominal_parameters)
        nominal_clear, _ = generate_clear_reference(
            self.config, pupil_amplitude=nominal_amplitude
        )
        if np.any(np.asarray(nominal_clear.shape) < np.asarray(measured_clear.shape)):
            raise ValueError(
                "The full theoretical detector image must be at least as large as "
                "the measured clear-pupil image."
            )

        measured_properties = measure_pupil(self.measured_clear)
        model_properties = measure_pupil(nominal_clear)
        measured_subpixel_center = estimate_pupil_center_subpixel(self.measured_clear)
        model_subpixel_center = estimate_pupil_center_subpixel(nominal_clear)
        self.measured_properties = measured_properties
        self.measured_subpixel_center = measured_subpixel_center
        self.model_subpixel_center = model_subpixel_center
        if nominal_clear.shape == self.measured_clear.shape:
            # Preserve absolute detector registration when a full frame is
            # supplied, so a physical pupil translation remains observable.
            self.crop_origin_yx = (0.0, 0.0)
            self.full_frame_registration = True
        else:
            # A cropped acquisition does not retain its full-frame origin.
            # Register pupil centres and infer rotation/scale from morphology.
            self.crop_origin_yx = (
                model_subpixel_center["center_y"]
                - measured_subpixel_center["center_y"],
                model_subpixel_center["center_x"]
                - measured_subpixel_center["center_x"],
            )
            self.full_frame_registration = False
        dilation = int(self.settings.get("fit_mask_dilation", 2))
        self.fit_mask = ndimage.binary_dilation(
            measured_properties["support"], iterations=max(dilation, 0)
        )
        self.measured_signal = np.clip(
            self.measured_clear - measured_properties["background"], 0.0, None
        )
        measured_mean = float(np.mean(self.measured_signal[self.fit_mask]))
        if measured_mean <= 0:
            raise ValueError("The measured clear pupil has no positive signal.")
        self.measured_normalized = self.measured_signal / measured_mean
        self.cache = {}

    def amplitude_from_parameters(self, parameters: np.ndarray) -> np.ndarray:
        parameters = np.asarray(parameters, dtype=float).copy()
        if self.circular_prior:
            parameters[0] = 0.0
        return transform_pupil(self.prior_pupil, *parameters)

    def model_patch(self, amplitude: np.ndarray) -> np.ndarray:
        clear, _ = generate_clear_reference(
            self.config, pupil_amplitude=amplitude
        )
        return _registered_extract(
            clear,
            self.measured_clear.shape,
            self.crop_origin_yx,
            (0.0, 0.0),
        )

    def residual_for_amplitude(self, amplitude: np.ndarray) -> np.ndarray:
        patch = self.model_patch(amplitude)
        model_mean = float(np.mean(patch[self.fit_mask]))
        if not np.isfinite(model_mean) or model_mean <= 0:
            return np.full(np.count_nonzero(self.fit_mask), 1e6)
        normalized = patch / model_mean
        return normalized[self.fit_mask] - self.measured_normalized[self.fit_mask]

    def residual(self, parameters: np.ndarray) -> np.ndarray:
        key = tuple(np.asarray(parameters, dtype=float))
        if key not in self.cache:
            amplitude = self.amplitude_from_parameters(parameters)
            self.cache[key] = self.residual_for_amplitude(amplitude)
        return self.cache[key]

    def cost(self, parameters: np.ndarray) -> float:
        residual = self.residual(parameters)
        return float(np.mean(residual**2))


def _fit_smooth_correction(
    objective: PupilObjective,
    geometric_amplitude: np.ndarray,
    settings: dict,
):
    grid_shape = tuple(map(int, settings.get("amplitude_grid_shape", [4, 4])))
    if len(grid_shape) != 2 or min(grid_shape) < 2:
        raise ValueError("pupil_fit.amplitude_grid_shape must contain two values >= 2.")
    coefficients = np.zeros(grid_shape, dtype=float)
    coefficient_limit = float(settings.get("amplitude_coefficient_limit", 0.5))
    epsilon = float(settings.get("amplitude_finite_difference", 0.03))
    prior_weight = float(settings.get("amplitude_prior_weight", 0.05))
    smooth_weight = float(settings.get("amplitude_smoothness_weight", 0.2))
    passes = int(settings.get("amplitude_passes", 1))
    difference = _difference_matrix(grid_shape)

    def amplitude_at(values):
        correction = correction_from_grid(values, geometric_amplitude.shape)
        return geometric_amplitude * np.clip(
            1.0 + correction,
            1.0 - coefficient_limit,
            1.0 + coefficient_limit,
        )

    for _ in range(max(passes, 0)):
        residual = objective.residual_for_amplitude(amplitude_at(coefficients))
        jacobian = np.empty((residual.size, coefficients.size))
        for index in range(coefficients.size):
            trial = coefficients.copy().ravel()
            trial[index] += epsilon
            trial = trial.reshape(grid_shape)
            trial_residual = objective.residual_for_amplitude(amplitude_at(trial))
            jacobian[:, index] = (trial_residual - residual) / epsilon

        flat = coefficients.ravel()
        blocks = [jacobian]
        targets = [-residual]
        if prior_weight > 0:
            blocks.append(np.sqrt(prior_weight) * np.eye(flat.size))
            targets.append(-np.sqrt(prior_weight) * flat)
        if smooth_weight > 0 and difference.size:
            blocks.append(np.sqrt(smooth_weight) * difference)
            targets.append(-np.sqrt(smooth_weight) * difference @ flat)
        step = np.linalg.lstsq(
            np.vstack(blocks), np.concatenate(targets), rcond=None
        )[0]

        original_cost = float(np.mean(residual**2))
        accepted = flat
        for fraction in (1.0, 0.5, 0.25, 0.1):
            candidate = np.clip(
                flat + fraction * step, -coefficient_limit, coefficient_limit
            )
            # Overall amplitude is carried by the independently fitted flux
            # scale, so retain only spatial (zero-mean) illumination changes.
            candidate -= np.mean(candidate)
            candidate = np.clip(candidate, -coefficient_limit, coefficient_limit)
            candidate_cost = float(np.mean(objective.residual_for_amplitude(
                amplitude_at(candidate.reshape(grid_shape))
            ) ** 2))
            if candidate_cost < original_cost:
                accepted = candidate
                break
        coefficients = accepted.reshape(grid_shape)

    correction = correction_from_grid(coefficients, geometric_amplitude.shape)
    fitted_amplitude = amplitude_at(coefficients)
    return fitted_amplitude, correction, coefficients


def fit_pupil_amplitude(
    measured_clear: np.ndarray,
    config: dict,
    output_directory: str | Path,
    write_diagnostics: bool = True,
) -> dict:
    """Fit pupil geometry and smooth amplitude, then generate updated references."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    material_path = Path(__file__).with_name("Exposed_Ma-N_1405_optical_constants.txt")
    settings = config.get("pupil_fit", {})
    objective = PupilObjective(measured_clear, config)

    nominal_rotation = float(config["pupil"].get("rotation_deg", 0.0))
    rotation_bounds = settings.get(
        "rotation_bounds_deg", [nominal_rotation - 15.0, nominal_rotation + 15.0]
    )
    shift_x_bounds = settings.get("shift_x_bounds_pixels", [-4.0, 4.0])
    shift_y_bounds = settings.get("shift_y_bounds_pixels", [-4.0, 4.0])
    scale_bounds = settings.get("diameter_scale_bounds", [0.95, 1.05])
    bounds = np.asarray(
        [rotation_bounds, shift_x_bounds, shift_y_bounds, scale_bounds], dtype=float
    )
    if bounds.shape != (4, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Invalid pupil_fit geometric parameter bounds.")

    rotation_values = np.linspace(
        bounds[0, 0], bounds[0, 1],
        int(settings.get("rotation_coarse_points", 13)),
    )
    rotation_costs = np.empty(rotation_values.size)
    initial = np.array([nominal_rotation, 0.0, 0.0, 1.0])
    initial = np.clip(initial, bounds[:, 0], bounds[:, 1])
    if objective.circular_prior:
        rotation_costs.fill(objective.cost(initial))
        rotation_observable = False
    else:
        for index, rotation in enumerate(rotation_values):
            trial = initial.copy()
            trial[0] = rotation
            rotation_costs[index] = objective.cost(trial)
        rotation_observable = not np.allclose(
            rotation_costs, rotation_costs[0], rtol=1e-5, atol=1e-10
        )
    if rotation_observable:
        initial[0] = rotation_values[np.argmin(rotation_costs)]

    geometric_result = least_squares(
        objective.residual,
        initial,
        bounds=(bounds[:, 0], bounds[:, 1]),
        x_scale=np.asarray(settings.get(
            "geometric_parameter_scales", [1.0, 0.5, 0.5, 0.01]
        ), dtype=float),
        max_nfev=int(settings.get("geometric_max_function_evaluations", 30)),
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7,
    )
    if not rotation_observable:
        geometric_result.x[0] = nominal_rotation
    geometric_amplitude = objective.amplitude_from_parameters(geometric_result.x)

    if bool(settings.get("fit_smooth_amplitude", True)):
        fitted_amplitude, correction, coefficients = _fit_smooth_correction(
            objective, geometric_amplitude, settings
        )
    else:
        fitted_amplitude = geometric_amplitude
        correction = np.zeros_like(geometric_amplitude)
        coefficients = np.zeros((0, 0))

    model_config = copy.deepcopy(config)
    model_config.setdefault("detector", {})["crop"] = None
    full_clear, full_masked, wavelengths = generate_references(
        model_config, material_path, pupil_amplitude=fitted_amplitude
    )
    model_clear = _registered_extract(
        full_clear, measured_clear.shape, objective.crop_origin_yx, (0.0, 0.0)
    )
    model_masked = _registered_extract(
        full_masked, measured_clear.shape, objective.crop_origin_yx, (0.0, 0.0)
    )
    flux_scale, background = _solve_photometry(
        model_clear, measured_clear, objective.fit_mask
    )
    fitted_clear = flux_scale * model_clear + background
    predicted_masked = flux_scale * model_masked + background
    residual_clear = measured_clear - fitted_clear

    fitted_config = copy.deepcopy(config)
    fitted_config["pupil"]["rotation_deg"] = float(geometric_result.x[0])
    fitted_config["pupil_fit_result"] = {
        "shift_x_pupil_pixels": float(geometric_result.x[1]),
        "shift_y_pupil_pixels": float(geometric_result.x[2]),
        "diameter_scale": float(geometric_result.x[3]),
        "amplitude_file": "pupil_fit_products.fits[PUPIL_AMPLITUDE]",
    }

    summary = {
        "parameter_names": list(PARAMETER_NAMES),
        "initial_parameters": [nominal_rotation, 0.0, 0.0, 1.0],
        "fitted_parameters": geometric_result.x.tolist(),
        "bounds": bounds.tolist(),
        "fit_success": bool(geometric_result.success),
        "fit_message": str(geometric_result.message),
        "nfev": int(geometric_result.nfev),
        "geometric_mean_squared_residual": objective.cost(geometric_result.x),
        "final_mean_squared_residual": float(np.mean(
            objective.residual_for_amplitude(fitted_amplitude) ** 2
        )),
        "flux_scale": flux_scale,
        "background": background,
        "crop_origin_yx": list(objective.crop_origin_yx),
        "rotation_observable": bool(rotation_observable),
        "full_frame_registration": bool(objective.full_frame_registration),
        "measured_pupil": {
            key: value for key, value in objective.measured_properties.items()
            if key != "support"
        },
        "measured_pupil_center_subpixel": objective.measured_subpixel_center,
        "model_pupil_center_subpixel": objective.model_subpixel_center,
        "amplitude_grid_coefficients": coefficients.tolist(),
        "config": fitted_config,
    }
    (output_directory / "pupil_fit_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    (output_directory / "fitted_config.json").write_text(
        json.dumps(fitted_config, indent=2)
    )

    product_header = fits.Header()
    product_header["BUNIT"] = "dimensionless amplitude"
    product_header["EXTNAME"] = "PUPIL_AMPLITUDE"
    product_header["PUPROT"] = float(geometric_result.x[0])
    product_header["PUPSHX"] = float(geometric_result.x[1])
    product_header["PUPSHY"] = float(geometric_result.x[2])
    product_header["PUPSCALE"] = float(geometric_result.x[3])
    product_header["PUPXCEN"] = (
        objective.measured_subpixel_center["center_x"],
        "Measured pupil X centre [zero-indexed pixel]",
    )
    product_header["PUPYCEN"] = (
        objective.measured_subpixel_center["center_y"],
        "Measured pupil Y centre [zero-indexed pixel]",
    )
    fits.HDUList([
        fits.PrimaryHDU(fitted_amplitude.astype(np.float64), product_header),
        fits.ImageHDU(objective.prior_pupil.astype(np.float64), name="PUPIL_PRIOR"),
        fits.ImageHDU(geometric_amplitude.astype(np.float64), name="GEOMETRIC_PUPIL"),
        fits.ImageHDU(correction.astype(np.float64), name="AMPLITUDE_CORRECTION"),
        fits.ImageHDU(measured_clear.astype(np.float64), name="MEASURED_CLEAR"),
        fits.ImageHDU(fitted_clear.astype(np.float64), name="MODEL_CLEAR"),
        fits.ImageHDU(residual_clear.astype(np.float64), name="RESID_CLEAR"),
        fits.ImageHDU(predicted_masked.astype(np.float64), name="PREDICTED_ZWFS"),
        fits.ImageHDU(objective.fit_mask.astype(np.uint8), name="FIT_MASK"),
    ]).writeto(output_directory / "pupil_fit_products.fits", overwrite=True)

    reference_header = fits.Header()
    reference_header["BUNIT"] = ("electron", "Expected photoelectrons per exposure")
    reference_header["EXTNAME"] = "CLEAR_PUPIL"
    reference_header["PUPROT"] = float(geometric_result.x[0])
    reference_header["PUPSHX"] = float(geometric_result.x[1])
    reference_header["PUPSHY"] = float(geometric_result.x[2])
    reference_header["PUPSCALE"] = float(geometric_result.x[3])
    reference_header["PUPXCEN"] = (
        objective.measured_subpixel_center["center_x"],
        "Measured pupil X centre [zero-indexed pixel]",
    )
    reference_header["PUPYCEN"] = (
        objective.measured_subpixel_center["center_y"],
        "Measured pupil Y centre [zero-indexed pixel]",
    )
    reference_header["NWAVE"] = len(wavelengths)
    add_complete_config(reference_header, fitted_config)
    masked_header = reference_header.copy()
    masked_header["EXTNAME"] = "PHASE_MASK"
    fits.HDUList([
        fits.PrimaryHDU(fitted_clear.astype(np.float64), reference_header),
        fits.ImageHDU(
            predicted_masked.astype(np.float64), masked_header, name="PHASE_MASK"
        ),
    ]).writeto(output_directory / "updated_references.fits", overwrite=True)

    if write_diagnostics:
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
        panels = (
            (objective.prior_pupil, "Configured pupil prior", "viridis"),
            (geometric_amplitude, "Registered geometric pupil", "viridis"),
            (correction, "Smooth amplitude correction", "RdBu_r"),
            (fitted_amplitude, "Fitted pupil amplitude", "viridis"),
            (measured_clear, "Measured on-sky clear pupil", "viridis"),
            (fitted_clear, "Forward-modelled clear pupil", "viridis"),
            (residual_clear, "Clear-pupil residual", "RdBu_r"),
            (predicted_masked, "Predicted on-sky ZWFS reference", "viridis"),
        )
        for axis, (image, title, cmap) in zip(axes.flat, panels):
            kwargs = {}
            if cmap == "RdBu_r":
                limit = float(np.max(np.abs(image)))
                if limit > 0:
                    kwargs = {"vmin": -limit, "vmax": limit}
            shown = axis.imshow(image, origin="lower", cmap=cmap, **kwargs)
            axis.set_title(title)
            axis.set_xlabel("x [pixel]")
            axis.set_ylabel("y [pixel]")
            fig.colorbar(shown, ax=axis, fraction=0.046)
        fig.savefig(output_directory / "pupil_fit_diagnostics.png", dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
        axis.plot(rotation_values, rotation_costs, "o-")
        axis.axvline(geometric_result.x[0], color="tab:red", linestyle="--")
        axis.set_xlabel("Absolute pupil rotation [deg]")
        axis.set_ylabel("Mean squared normalized residual")
        axis.set_title("Coarse pupil-rotation search")
        fig.savefig(output_directory / "pupil_rotation_scan.png", dpi=180)
        plt.close(fig)

        centre = objective.measured_subpixel_center
        fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
        shown = axis.imshow(measured_clear, origin="lower", cmap="viridis")
        axis.add_patch(plt.Circle(
            (centre["center_x"], centre["center_y"]),
            centre["radius_pixels"],
            fill=False,
            color="white",
            linewidth=1.5,
        ))
        axis.plot(
            centre["center_x"], centre["center_y"], "+",
            color="tab:red", markersize=12, markeredgewidth=1.5,
        )
        axis.set_xlabel("x [zero-indexed pixel]")
        axis.set_ylabel("y [zero-indexed pixel]")
        axis.set_title(
            "Outer-edge subpixel centre\n"
            f"x={centre['center_x']:.4f}, y={centre['center_y']:.4f}; "
            f"coverage={centre['angular_coverage_fraction']:.0%}"
        )
        fig.colorbar(shown, ax=axis, fraction=0.046)
        fig.savefig(output_directory / "pupil_center_fit.png", dpi=180)
        plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "references", type=Path,
        help="FITS containing the measured clear pupil (and optionally ZWFS pupil).",
    )
    parser.add_argument("config", type=Path, help="Model and pupil-fit JSON config.")
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    measured_clear = read_clear_fits(args.references)
    summary = fit_pupil_amplitude(
        measured_clear, load_config(args.config), args.output_directory
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
