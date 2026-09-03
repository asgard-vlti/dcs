#!/usr/bin/env python3
"""Visualize automatic pupil registration and theoretical-model cropping."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from astropy.io import fits

from .fitting import (
    AlignmentObjective,
    _registered_extract,
    measure_pupil,
    parameters_from_config,
    read_reference_fits,
)
from .model import generate_references, load_config


def _normalise(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def visualize_registration(
    config: dict,
    output_directory: str | Path,
    references: str | Path | None = None,
    crop_size: int = 32,
    offset_x: float = 3.0,
    offset_y: float = -2.0,
    noise_fraction: float = 0.0,
    seed: int = 4,
) -> dict:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    material = Path(__file__).with_name("Exposed_Ma-N_1405_optical_constants.txt")

    # Always retain the complete theoretical detector frame for registration.
    model_config = copy.deepcopy(config)
    model_config.setdefault("detector", {})["crop"] = None
    model_config.setdefault("fit", {})["registration"] = "auto"
    full_clear, full_masked, _ = generate_references(model_config, material)
    full_measurement = measure_pupil(full_clear)
    full_center = (full_measurement["center_y"], full_measurement["center_x"])

    true_origin = None
    requested_center = None
    if references is None:
        shape = (int(crop_size), int(crop_size))
        requested_center = (
            (shape[0] - 1) / 2 + float(offset_y),
            (shape[1] - 1) / 2 + float(offset_x),
        )
        true_origin = (
            full_center[0] - requested_center[0],
            full_center[1] - requested_center[1],
        )
        measured_clear = _registered_extract(full_clear, shape, full_center, requested_center)
        measured_masked = _registered_extract(full_masked, shape, full_center, requested_center)
        if noise_fraction > 0:
            rng = np.random.default_rng(seed)
            measured_clear += rng.normal(0, noise_fraction * measured_clear.max(), shape)
            measured_masked += rng.normal(0, noise_fraction * measured_masked.max(), shape)
        mode = "synthetic"
    else:
        measured_clear, measured_masked = read_reference_fits(references)
        mode = "measured"

    objective = AlignmentObjective(measured_clear, measured_masked, model_config, material)
    registered_clear, registered_masked, photometry = objective.model(
        parameters_from_config(model_config)
    )
    residual_clear = measured_clear - registered_clear
    residual_masked = measured_masked - registered_masked
    measured_center = (
        objective.measurement["center_y"], objective.measurement["center_x"]
    )
    origin = objective.crop_origin_yx

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), constrained_layout=True)
    panels = (
        (full_clear, "Full theoretical clear"),
        (measured_clear, "Input clear subframe"),
        (registered_clear, "Registered/cropped clear model"),
        (residual_clear, "Clear residual"),
        (full_masked, "Full theoretical ZWFS"),
        (measured_masked, "Input ZWFS subframe"),
        (registered_masked, "Registered/cropped ZWFS model"),
        (residual_masked, "ZWFS residual"),
    )
    for axis, (panel, title) in zip(axes.flat, panels):
        is_residual = "residual" in title.lower()
        shown = panel if is_residual else _normalise(panel)
        limit = float(np.max(np.abs(shown))) if is_residual else 0.0
        kwargs = {"vmin": -limit, "vmax": limit} if limit > 0 else {}
        view = axis.imshow(
            shown, origin="lower", cmap="RdBu_r" if is_residual else "viridis", **kwargs
        )
        axis.set_title(title)
        axis.set_xlabel("detector x [pixel]")
        axis.set_ylabel("detector y [pixel]")
        fig.colorbar(view, ax=axis, fraction=0.046)

    for axis in (axes[0, 0], axes[1, 0]):
        axis.add_patch(Rectangle(
            (origin[1] - 0.5, origin[0] - 0.5),
            measured_clear.shape[1], measured_clear.shape[0],
            fill=False, color="white", linewidth=1.8,
        ))
        axis.plot(full_center[1], full_center[0], "+", color="cyan", markersize=12)
    for axis in (axes[0, 1], axes[1, 1], axes[0, 2], axes[1, 2]):
        axis.plot(measured_center[1], measured_center[0], "+", color="red", markersize=12)

    figure_path = output_directory / "registration_and_crop.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    fits.HDUList([
        fits.PrimaryHDU(measured_clear),
        fits.ImageHDU(measured_masked, name="MEASURED_ZWFS"),
        fits.ImageHDU(registered_clear, name="CROPPED_CLEAR"),
        fits.ImageHDU(registered_masked, name="CROPPED_ZWFS"),
        fits.ImageHDU(residual_clear, name="RESID_CLEAR"),
        fits.ImageHDU(residual_masked, name="RESID_ZWFS"),
    ]).writeto(output_directory / "registration_products.fits", overwrite=True)

    report = {
        "mode": mode,
        "full_model_shape": list(full_clear.shape),
        "measured_shape": list(measured_clear.shape),
        "full_model_pupil_center_yx": list(full_center),
        "detected_measured_pupil_center_yx": list(measured_center),
        "recovered_model_crop_origin_yx": list(origin),
        "synthetic_true_crop_origin_yx": list(true_origin) if true_origin else None,
        "synthetic_requested_pupil_center_yx": list(requested_center) if requested_center else None,
        "flux_scale": photometry[0],
        "clear_background": photometry[1],
        "zwfs_background": photometry[2],
        "clear_relative_rms": float(np.std(residual_clear) / max(np.std(measured_clear), 1e-30)),
        "zwfs_relative_rms": float(np.std(residual_masked) / max(np.std(measured_masked), 1e-30)),
    }
    (output_directory / "registration_summary.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Model/fit JSON configuration")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--references", type=Path, help="Optional measured clear/ZWFS FITS")
    parser.add_argument("--crop-size", type=int, default=32, help="Synthetic square subframe size")
    parser.add_argument("--offset-x", type=float, default=3.0, help="Synthetic pupil offset in pixels")
    parser.add_argument("--offset-y", type=float, default=-2.0, help="Synthetic pupil offset in pixels")
    parser.add_argument("--noise-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=4)
    args = parser.parse_args()
    report = visualize_registration(
        load_config(args.config), args.output_directory, args.references,
        args.crop_size, args.offset_x, args.offset_y, args.noise_fraction, args.seed,
    )
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output_directory / 'registration_and_crop.png'}")


if __name__ == "__main__":
    main()
