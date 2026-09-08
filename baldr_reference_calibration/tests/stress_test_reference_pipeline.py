#!/usr/bin/env python3
"""Stress-test internal alignment fitting and the on-sky reference update."""

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from baldr_reference.fitting import config_with_parameters, fit_alignment
from baldr_reference.model import generate_references, load_config, make_pupil
from baldr_reference.pupil_fitting import (
    correction_from_grid,
    fit_pupil_amplitude,
    transform_pupil,
)


parser = argparse.ArgumentParser(description=__doc__)
repo_root = Path(__file__).resolve().parents[1]
parser.add_argument(
    "--internal-config",
    type=Path,
    default=repo_root / "configs" / "stress_internal.example.json",
)
parser.add_argument(
    "--onsky-config",
    type=Path,
    default=repo_root / "configs" / "stress_onsky.example.json",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("stress_test_reference_pipeline_output"),
)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
internal_output = args.output_dir / "01_internal_alignment_fit"
onsky_output = args.output_dir / "02_onsky_pupil_fit"
material_path = (
    repo_root
    / "src"
    / "baldr_reference"
    / "Exposed_Ma-N_1405_optical_constants.txt"
)

internal_config = load_config(args.internal_config)
onsky_config = load_config(args.onsky_config)


# -----------------------------------------------------------------------------
# 1. Generate noisy internal Solarstein references with known relay misalignment.
# -----------------------------------------------------------------------------

internal_settings = internal_config["stress_test"]
true_alignment = np.asarray(
    internal_settings["true_alignment_parameters"], dtype=float
)
true_internal_config = config_with_parameters(internal_config, true_alignment)
true_internal_clear, true_internal_zwfs, _ = generate_references(
    true_internal_config, material_path
)

rng = np.random.default_rng(int(internal_settings["random_seed"]))
internal_noise = float(internal_settings["noise_fraction"])
measured_internal_clear = true_internal_clear + rng.normal(
    0.0, internal_noise * true_internal_clear.max(), true_internal_clear.shape
)
measured_internal_zwfs = true_internal_zwfs + rng.normal(
    0.0, internal_noise * true_internal_zwfs.max(), true_internal_zwfs.shape
)

fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(measured_internal_clear, name="N0"),
    fits.ImageHDU(measured_internal_zwfs, name="I0"),
]).writeto(args.output_dir / "synthetic_internal_measurement.fits", overwrite=True)

internal_summary = fit_alignment(
    measured_internal_clear,
    measured_internal_zwfs,
    internal_config,
    internal_output,
)
fitted_alignment = np.asarray(
    internal_summary["fitted_parameters"], dtype=float
)


# -----------------------------------------------------------------------------
# 2. Generate a noisy on-sky clear pupil using the true relay alignment.
# -----------------------------------------------------------------------------

onsky_settings = onsky_config["stress_test"]
truth_rotation = float(onsky_config["pupil"]["rotation_deg"])
truth_pupil_parameters = np.array([
    truth_rotation,
    float(onsky_settings["true_shift_x_pupil_pixels"]),
    float(onsky_settings["true_shift_y_pupil_pixels"]),
    float(onsky_settings["true_diameter_scale"]),
])

unrotated_config = copy.deepcopy(onsky_config)
unrotated_config["pupil"]["rotation_deg"] = 0.0
prior_onsky_pupil = make_pupil(unrotated_config)
true_geometric_pupil = transform_pupil(
    prior_onsky_pupil, *truth_pupil_parameters
)
true_coefficients = np.asarray(
    onsky_settings["true_amplitude_grid_coefficients"], dtype=float
)
true_correction = correction_from_grid(
    true_coefficients, true_geometric_pupil.shape
)
true_onsky_amplitude = true_geometric_pupil * (1.0 + true_correction)

true_onsky_config = config_with_parameters(onsky_config, true_alignment)
true_onsky_clear, true_onsky_zwfs, _ = generate_references(
    true_onsky_config,
    material_path,
    pupil_amplitude=true_onsky_amplitude,
)

rng = np.random.default_rng(int(onsky_settings["random_seed"]))
onsky_noise = float(onsky_settings["noise_fraction"])
measured_onsky_clear = true_onsky_clear + rng.normal(
    0.0, onsky_noise * true_onsky_clear.max(), true_onsky_clear.shape
)

fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(measured_onsky_clear, name="N0"),
    fits.ImageHDU(true_onsky_zwfs, name="I0_TRUTH"),
]).writeto(args.output_dir / "synthetic_onsky_measurement.fits", overwrite=True)


# -----------------------------------------------------------------------------
# 3. Carry the recovered internal alignment into the on-sky pupil fit.
# -----------------------------------------------------------------------------

onsky_fit_config = config_with_parameters(onsky_config, fitted_alignment)
onsky_fit_config["pupil"]["rotation_deg"] = float(
    onsky_settings["initial_rotation_deg"]
)
onsky_summary = fit_pupil_amplitude(
    measured_onsky_clear,
    onsky_fit_config,
    onsky_output,
)
fitted_pupil_parameters = np.asarray(
    onsky_summary["fitted_parameters"], dtype=float
)

with fits.open(onsky_output / "pupil_fit_products.fits") as hdus:
    fitted_amplitude = np.asarray(hdus["PUPIL_AMPLITUDE"].data, dtype=float)
    fitted_onsky_clear = np.asarray(hdus["MODEL_CLEAR"].data, dtype=float)
    predicted_onsky_zwfs = np.asarray(hdus["PREDICTED_ZWFS"].data, dtype=float)


# -----------------------------------------------------------------------------
# 4. Quantify recovery and create a combined end-to-end diagnostic.
# -----------------------------------------------------------------------------

def relative_l2(model, truth):
    return float(np.linalg.norm(model - truth) / max(np.linalg.norm(truth), 1e-30))


amplitude_scale = np.sum(true_onsky_amplitude * fitted_amplitude) / max(
    np.sum(fitted_amplitude**2), 1e-30
)
scaled_fitted_amplitude = amplitude_scale * fitted_amplitude

with fits.open(internal_output / "fit_products.fits") as hdus:
    fitted_internal_clear = np.asarray(hdus["MODEL_CLEAR"].data, dtype=float)
    fitted_internal_zwfs = np.asarray(hdus["MODEL_ZWFS"].data, dtype=float)

summary = {
    "internal": {
        "true_alignment_parameters": true_alignment.tolist(),
        "fitted_alignment_parameters": fitted_alignment.tolist(),
        "alignment_error": (fitted_alignment - true_alignment).tolist(),
        "parameter_names": internal_summary["parameter_names"],
    },
    "onsky": {
        "spectral_type": onsky_config["source"]["profiles"]["onsky"]["spectral_type"],
        "strehl": float(onsky_config["optics"]["strehl"]),
        "true_pupil_parameters": truth_pupil_parameters.tolist(),
        "fitted_pupil_parameters": fitted_pupil_parameters.tolist(),
        "pupil_parameter_error": (
            fitted_pupil_parameters - truth_pupil_parameters
        ).tolist(),
        "parameter_names": onsky_summary["parameter_names"],
        "clear_relative_l2_error": relative_l2(
            fitted_onsky_clear, true_onsky_clear
        ),
        "amplitude_relative_l2_error_after_scalar_matching": relative_l2(
            scaled_fitted_amplitude, true_onsky_amplitude
        ),
        "zwfs_relative_l2_error": relative_l2(
            predicted_onsky_zwfs, true_onsky_zwfs
        ),
        "geometric_mean_squared_residual": onsky_summary[
            "geometric_mean_squared_residual"
        ],
        "final_mean_squared_residual": onsky_summary[
            "final_mean_squared_residual"
        ],
    },
}
(args.output_dir / "stress_test_summary.json").write_text(
    json.dumps(summary, indent=2)
)

fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
panels = (
    (measured_internal_clear, "Internal measured clear", "viridis"),
    (fitted_internal_clear, "Internal fitted clear", "viridis"),
    (measured_internal_clear - fitted_internal_clear, "Internal clear residual", "RdBu_r"),
    (measured_internal_zwfs - fitted_internal_zwfs, "Internal ZWFS residual", "RdBu_r"),
    (true_onsky_amplitude, "True on-sky amplitude", "viridis"),
    (scaled_fitted_amplitude, "Recovered on-sky amplitude", "viridis"),
    (scaled_fitted_amplitude - true_onsky_amplitude, "Amplitude error", "RdBu_r"),
    (measured_onsky_clear, "Measured on-sky clear", "viridis"),
    (true_onsky_clear, "True on-sky clear", "viridis"),
    (fitted_onsky_clear, "Updated clear reference", "viridis"),
    (true_onsky_zwfs, "True on-sky ZWFS", "viridis"),
    (predicted_onsky_zwfs - true_onsky_zwfs, "Updated ZWFS error", "RdBu_r"),
)
for axis, (image, title, cmap) in zip(axes.flat, panels):
    kwargs = {}
    if cmap == "RdBu_r":
        limit = float(np.max(np.abs(image)))
        if limit > 0:
            kwargs = {"vmin": -limit, "vmax": limit}
    artist = axis.imshow(image, origin="lower", cmap=cmap, **kwargs)
    axis.set_title(title)
    axis.set_xlabel("x [pixel]")
    axis.set_ylabel("y [pixel]")
    fig.colorbar(artist, ax=axis, fraction=0.046)
fig.savefig(args.output_dir / "stress_test_pipeline.png", dpi=100)
plt.close(fig)

print("\nInternal alignment recovery")
for name, truth, fitted in zip(
    internal_summary["parameter_names"], true_alignment, fitted_alignment
):
    print(f"  {name:24s} truth={truth:+10.4f}  fitted={fitted:+10.4f}")

print("\nOn-sky pupil recovery")
for name, truth, fitted in zip(
    onsky_summary["parameter_names"],
    truth_pupil_parameters,
    fitted_pupil_parameters,
):
    print(f"  {name:28s} truth={truth:+10.4f}  fitted={fitted:+10.4f}")

print("\nReference errors")
print(f"  clear relative L2:     {summary['onsky']['clear_relative_l2_error']:.6e}")
print(f"  amplitude relative L2: {summary['onsky']['amplitude_relative_l2_error_after_scalar_matching']:.6e}")
print(f"  ZWFS relative L2:      {summary['onsky']['zwfs_relative_l2_error']:.6e}")
print(f"\nWrote outputs to {args.output_dir.resolve()}")
