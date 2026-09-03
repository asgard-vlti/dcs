#!/usr/bin/env python3
"""Stress-test alignment recovery against noise and model uncertainties."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import (
    PHASE_MASK_PRESETS,
    generate_references,
    load_config,
    resolve_source_profile,
)
from .fitting import PARAMETER_NAMES, config_with_parameters, fit_alignment


PARAMETER_UNITS = ("mm", "deg", "um", "um")


def perturb_truth(config: dict, perturbation: dict) -> dict:
    result = copy.deepcopy(config)
    mask = result["phase_mask"]
    mask_name = mask.get("name")
    if mask_name:
        mask.update(PHASE_MASK_PRESETS[str(mask_name).upper()])
        mask.update(
            name=None,
            phase_model="physical_depth",
            diameter_model="physical",
            material="N_1405",
        )
    mask["diameter_um"] = float(mask["diameter_um"]) + float(
        perturbation.get("mask_diameter_delta_um", 0.0)
    )
    mask["depth_um"] = float(mask["depth_um"]) + float(
        perturbation.get("mask_depth_delta_um", 0.0)
    )

    source_name, _, nominal_temperature, _ = resolve_source_profile(result)
    profile = result["source"]["profiles"][source_name]
    profile["temperature_K"] = nominal_temperature + float(
        perturbation.get("temperature_delta_K", 0.0)
    )

    spectrum = result["spectrum"]
    minimum = float(spectrum["wavelength_min_m"])
    maximum = float(spectrum["wavelength_max_m"])
    center = 0.5 * (minimum + maximum)
    width = maximum - minimum
    center += float(perturbation.get("passband_center_delta_nm", 0.0)) * 1e-9
    width *= 1.0 + float(perturbation.get("passband_width_fraction", 0.0))
    if width <= 0:
        raise ValueError("Perturbed passband width must remain positive.")
    spectrum["wavelength_min_m"] = center - width / 2
    spectrum["wavelength_max_m"] = center + width / 2
    return result


def summarize(rows):
    grouped = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == scenario]
        errors = np.asarray([[row[f"error_{name}"] for name in PARAMETER_NAMES] for row in selected])
        grouped[scenario] = {
            "runs": len(selected),
            "mean_error": dict(zip(PARAMETER_NAMES, errors.mean(axis=0).tolist())),
            "std_error": dict(zip(PARAMETER_NAMES, errors.std(axis=0).tolist())),
            "rmse": dict(zip(PARAMETER_NAMES, np.sqrt(np.mean(errors**2, axis=0)).tolist())),
            "all_successful": all(row["fit_success"] for row in selected),
        }
    return grouped


def plot_sensitivity(rows, output_path):
    scenario_names = list(dict.fromkeys(row["scenario"] for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(scenario_names) * 1.15), 8))
    for parameter_index, axis in enumerate(axes.flat):
        name = PARAMETER_NAMES[parameter_index]
        for scenario_index, scenario in enumerate(scenario_names):
            values = [row[f"error_{name}"] for row in rows if row["scenario"] == scenario]
            offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
            axis.scatter(np.asarray(offsets) + scenario_index, values, color="#4C78A8", zorder=3)
            axis.plot(
                [scenario_index - 0.2, scenario_index + 0.2],
                [np.mean(values), np.mean(values)], color="#F58518", linewidth=2,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(scenario_names)), scenario_names, rotation=45, ha="right")
        axis.set_ylabel(f"Fitted - true [{PARAMETER_UNITS[parameter_index]}]")
        axis.set_title(name)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Alignment sensitivity to measurement and model uncertainty")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config.get("sensitivity", {})
    scenarios = settings.get("scenarios")
    if not scenarios:
        raise ValueError("Configuration must contain sensitivity.scenarios.")
    truth = np.asarray(config["synthetic_test"]["true_parameters"], dtype=float)
    material = Path(__file__).with_name("Exposed_Ma-N_1405_optical_constants.txt")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    base_seed = int(settings.get("random_seed", 100))

    for scenario_index, scenario in enumerate(scenarios):
        scenario_name = str(scenario["name"])
        repeats = int(scenario.get("repeats", 1))
        perturbation = scenario.get("perturbation", {})
        truth_config = config_with_parameters(perturb_truth(config, perturbation), truth)
        noiseless_clear, noiseless_masked, _ = generate_references(truth_config, material)
        illuminated = noiseless_clear > 0.05 * noiseless_clear.max()
        signal_level = float(np.median(noiseless_clear[illuminated]))
        noise_std = float(perturbation.get("noise_fraction", 0.0)) * signal_level

        for repeat in range(repeats):
            rng = np.random.default_rng(base_seed + 1000 * scenario_index + repeat)
            clear = noiseless_clear + rng.normal(0, noise_std, noiseless_clear.shape)
            masked = noiseless_masked + rng.normal(0, noise_std, noiseless_masked.shape)
            run_directory = args.output_directory / "runs" / f"{scenario_index:02d}_{scenario_name}_{repeat:02d}"
            summary = fit_alignment(
                clear, masked, config, run_directory, write_diagnostics=False
            )
            fitted = np.asarray(summary["fitted_parameters"])
            row = {
                "scenario": scenario_name,
                "repeat": repeat,
                "noise_std_counts": noise_std,
                "fit_success": bool(summary["fit_success"]),
                "fit_cost": float(summary["cost"]),
            }
            for key, value in perturbation.items():
                row[f"perturb_{key}"] = value
            for index, name in enumerate(PARAMETER_NAMES):
                row[f"true_{name}"] = truth[index]
                row[f"fitted_{name}"] = fitted[index]
                row[f"error_{name}"] = fitted[index] - truth[index]
            rows.append(row)
            print(f"Completed {scenario_name} repeat {repeat + 1}/{repeats}")

    fieldnames = sorted({key for row in rows for key in row})
    with (args.output_directory / "sensitivity_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "parameter_names": list(PARAMETER_NAMES),
        "parameter_units": list(PARAMETER_UNITS),
        "true_parameters": truth.tolist(),
        "scenario_summary": summarize(rows),
        "config": config,
    }
    (args.output_directory / "sensitivity_summary.json").write_text(json.dumps(result, indent=2))
    plot_sensitivity(rows, args.output_directory / "sensitivity_alignment_errors.png")
    print(f"Wrote sensitivity analysis to {args.output_directory.resolve()}")


if __name__ == "__main__":
    main()
