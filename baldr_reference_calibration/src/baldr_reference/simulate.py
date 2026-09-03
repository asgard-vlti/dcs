#!/usr/bin/env python3
"""Generate a synthetic two-image reference and test alignment recovery."""

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

from .model import generate_references, load_config
from .fitting import PARAMETER_NAMES, config_with_parameters, fit_alignment


PARAMETER_LABELS = (
    ("Knife-edge offset", "mm"),
    ("Knife-edge angle", "deg"),
    ("Cold-stop x offset", "um"),
    ("Cold-stop y offset", "um"),
)


def plot_parameter_recovery(truth, fitted, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for index, (axis, (label, unit)) in enumerate(zip(axes.flat, PARAMETER_LABELS)):
        values = [truth[index], fitted[index]]
        axis.bar(["True", "Fitted"], values, color=["#4C78A8", "#F58518"])
        span = max(abs(values[0]), abs(values[1]), abs(values[1] - values[0]), 1e-9)
        padding = 0.18 * span
        axis.set_ylim(min(values + [0]) - padding, max(values + [0]) + padding)
        axis.set_ylabel(unit)
        axis.set_title(label)
        error = fitted[index] - truth[index]
        axis.text(
            0.5, 0.96, f"fitted - true = {error:+.5g} {unit}",
            transform=axis.transAxes, ha="center", va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Synthetic Baldr alignment parameter recovery")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    test = config["synthetic_test"]
    truth = np.asarray(test["true_parameters"], dtype=float)
    truth_config = config_with_parameters(config, truth)
    material = Path(__file__).with_name("Exposed_Ma-N_1405_optical_constants.txt")
    clear, masked, _ = generate_references(truth_config, material)
    rng = np.random.default_rng(int(test.get("random_seed", 5)))
    scale = float(test.get("flux_scale", 1.0))
    clear_background = float(test.get("clear_background", 0.0))
    masked_background = float(test.get("zwfs_background", 0.0))
    noise = float(test.get("noise_std", 0.0))
    clear = scale * clear + clear_background + rng.normal(0, noise, clear.shape)
    masked = scale * masked + masked_background + rng.normal(0, noise, masked.shape)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    synthetic_path = args.output_directory / "synthetic_references.fits"
    primary = fits.PrimaryHDU(clear)
    primary.header["EXTNAME"] = "CLEAR_PUPIL"
    fits.HDUList([
        primary,
        fits.ImageHDU(masked, name="PHASE_MASK"),
    ]).writeto(synthetic_path, overwrite=True)
    summary = fit_alignment(clear, masked, config, args.output_directory / "fit")
    fitted = np.asarray(summary["fitted_parameters"])
    report = {
        "parameter_names": list(PARAMETER_NAMES),
        "true_parameters": truth.tolist(),
        "fitted_parameters": fitted.tolist(),
        "errors": (fitted - truth).tolist(),
        "synthetic_references": str(synthetic_path),
    }
    (args.output_directory / "recovery_report.json").write_text(json.dumps(report, indent=2))
    plot_parameter_recovery(
        truth, fitted, args.output_directory / "parameter_recovery.png"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
