#!/usr/bin/env python3
"""Generate cropped Baldr reference FITS with command-line config overrides.

e.g.
python scripts/gen_refs.py \
    output/reference_H3.fits \
    --config configs/reference_onsky.example.json \
    --spectral-type K5V \
    --wavelength-min-um 1.50 \
    --wavelength-max-um 1.80 \
    --samples 7 \
    --pupil-geometry at \
    --strehl 0.8 \
    --phase-mask H3 \
    --edge-angle-deg 1.5 \
    --edge-offset-mm -1.0 \
    --cold-stop-x-um 50 \
    --cold-stop-y-um -25 \
    --frame-size 32 32 \
    --pupil-center 15.25 15.75 \
    --overwrite

"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

from baldr_reference.generate import add_complete_config
from baldr_reference.pupil_fitting import fit_pupil_amplitude
from baldr_reference.model import (
    generate_references,
    load_config,
    resolve_source_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "reference_onsky.example.json"


def acquire_clear_pupil(beam_id: int, frame_count: int, frame_sleep: float) -> tuple[np.ndarray, str]:
    """Average one clear-pupil sequence from the Baldr camera shared memory."""
    if frame_count < 1:
        raise ValueError("--n-clear must be at least 1.")
    if frame_sleep < 0.0:
        raise ValueError("--frame-sleep must be non-negative.")

    try:
        from xaosim.shmlib import shm
    except ImportError as error:
        raise RuntimeError(
            "Measurement mode requires xaosim and access to Baldr shared memory."
        ) from error

    shm_path = f"/dev/shm/baldr{beam_id}.im.shm"
    print(f"Opening shared memory: {shm_path}")
    camera = shm(shm_path)
    reference_shape = np.asarray(camera.get_data()).shape
    if len(reference_shape) != 2:
        raise RuntimeError(f"Expected a 2-D camera frame, got {reference_shape}.")

    print("Move the phase mask OUT of the beam and allow the pupil to settle.")
    input("Press Enter to acquire the clear pupil... ")
    total = np.zeros(reference_shape, dtype=np.float64)
    for index in range(frame_count):
        frame = np.asarray(camera.get_data(), dtype=np.float64)
        if frame.shape != reference_shape:
            raise RuntimeError(
                f"Camera shape changed from {reference_shape} to {frame.shape}."
            )
        total += frame
        if frame_sleep:
            time.sleep(frame_sleep)
        if index == 0 or (index + 1) % max(1, frame_count // 10) == 0:
            print(f"  clear pupil: {index + 1}/{frame_count}")
    return total / frame_count, shm_path


def clear_pupil_interior(image: np.ndarray, background: float = 0.0) -> np.ndarray:
    """Return illuminated pupil pixels excluding spiders, obstruction, and edge."""
    signal = np.clip(np.asarray(image, dtype=float) - float(background), 0.0, None)
    smooth = ndimage.gaussian_filter(signal, 0.7)
    peak = float(smooth.max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("The clear reference contains no positive pupil signal.")
    border = np.concatenate((signal[0], signal[-1], signal[:, 0], signal[:, -1]))
    mad = float(np.median(np.abs(border - np.median(border))))
    threshold = max(5.0 * 1.4826 * mad, 0.08 * peak)
    support = (smooth > threshold) & (
        signal > max(3.0 * 1.4826 * mad, 0.03 * peak)
    )

    labels, count = ndimage.label(support)
    if count:
        sizes = np.asarray(
            ndimage.sum(np.ones_like(labels), labels, range(1, count + 1)),
            dtype=float,
        )
        largest = float(sizes.max())
        retained = np.flatnonzero(sizes >= max(3.0, 0.08 * largest)) + 1
        support = np.isin(labels, retained)

    interior = ndimage.binary_erosion(support, iterations=1)
    if np.count_nonzero(interior) < 8:
        interior = support
    if np.count_nonzero(interior) < 8:
        raise ValueError("Too few clear-pupil interior pixels for normalization.")
    return interior


def normalize_reference_pair(
    clear: np.ndarray,
    masked: np.ndarray,
    clear_background: float = 0.0,
    masked_background: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Normalize both references by the clear-pupil interior mean signal."""
    clear = np.asarray(clear, dtype=float) - float(clear_background)
    masked = np.asarray(masked, dtype=float) - float(masked_background)
    if clear.shape != masked.shape:
        raise ValueError("Clear and ZWFS reference frames must have matching shapes.")
    interior = clear_pupil_interior(clear)
    normalization = float(np.mean(clear[interior]))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Clear-pupil normalization must be finite and positive.")
    return clear / normalization, masked / normalization, interior, normalization


def extract_frame(
    image: np.ndarray,
    frame_shape: tuple[int, int],
    pupil_center_xy: tuple[float, float],
) -> np.ndarray:
    """Place the full-frame centre at a floating-point output pixel position."""
    height, width = frame_shape
    pupil_center_x, pupil_center_y = pupil_center_xy
    full_center_y = (image.shape[0] - 1.0) / 2.0
    full_center_x = (image.shape[1] - 1.0) / 2.0

    rows, columns = np.indices((height, width), dtype=float)
    source_rows = rows + full_center_y - pupil_center_y
    source_columns = columns + full_center_x - pupil_center_x
    if (
        source_rows.min() < 0.0
        or source_rows.max() > image.shape[0] - 1.0
        or source_columns.min() < 0.0
        or source_columns.max() > image.shape[1] - 1.0
    ):
        raise ValueError(
            "The requested frame and pupil centre extend beyond the generated "
            f"{image.shape[1]}x{image.shape[0]} detector image."
        )

    return ndimage.map_coordinates(
        image,
        (source_rows, source_columns),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output reference FITS file")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Base JSON configuration (default: {DEFAULT_CONFIG})",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--spectral-type",
        help="Stellar spectral type, for example K5V; clears temperature_K",
    )
    source.add_argument(
        "--temperature-k",
        type=float,
        help="Blackbody temperature [K]; clears spectral_type",
    )

    parser.add_argument("--wavelength-min-um", type=float)
    parser.add_argument("--wavelength-max-um", type=float)
    parser.add_argument("--samples", type=int, help="Number of wavelength samples")
    parser.add_argument(
        "--pupil-geometry",
        type=str.lower,
        choices=("solarstein", "at", "ut", "disc"),
    )
    parser.add_argument("--strehl", type=float)
    parser.add_argument("--phase-mask", type=str.upper, choices=tuple(
        f"{band}{number}" for band in ("H", "J") for number in range(1, 6)
    ))
    parser.add_argument("--edge-angle-deg", type=float)
    parser.add_argument("--edge-offset-mm", type=float)
    parser.add_argument("--cold-stop-x-um", type=float)
    parser.add_argument("--cold-stop-y-um", type=float)

    parser.add_argument(
        "--measure-clear",
        action="store_true",
        help=(
            "Measure and fit the clear pupil from shared memory, then predict "
            "the ZWFS reference from the fitted pupil amplitude."
        ),
    )
    parser.add_argument(
        "--beam-id",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Baldr beam used by --measure-clear (default: 1)",
    )
    parser.add_argument(
        "--n-clear",
        type=int,
        default=500,
        help="Clear-pupil frames averaged by --measure-clear (default: 500)",
    )
    parser.add_argument(
        "--frame-sleep",
        type=float,
        default=0.01,
        help="Delay between shared-memory reads in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--pupil-fit-output-dir",
        type=Path,
        help="Optional directory for measured-pupil fit diagnostics",
    )

    parser.add_argument(
        "--frame-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=(32, 32),
        help="Output frame size in pixels (default: 32 32)",
    )
    parser.add_argument(
        "--pupil-center",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help=(
            "Desired pupil centre in zero-indexed output pixel coordinates. "
            "The default is the geometric frame centre, e.g. 15.5 15.5 for 32x32."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = copy.deepcopy(load_config(args.config))

    profile_name = str(config["source"]["profile"])
    profile = config["source"]["profiles"][profile_name]
    if args.spectral_type is not None:
        profile["spectral_type"] = args.spectral_type.strip().upper().replace(" ", "")
        profile["temperature_K"] = None
    elif args.temperature_k is not None:
        if not np.isfinite(args.temperature_k) or args.temperature_k <= 0.0:
            raise ValueError("--temperature-k must be finite and positive.")
        profile["temperature_K"] = float(args.temperature_k)
        profile["spectral_type"] = None

    spectrum = config["spectrum"]
    if args.wavelength_min_um is not None:
        spectrum["wavelength_min_m"] = float(args.wavelength_min_um) * 1e-6
    if args.wavelength_max_um is not None:
        spectrum["wavelength_max_m"] = float(args.wavelength_max_um) * 1e-6
    if args.samples is not None:
        if args.samples < 1:
            raise ValueError("--samples must be at least 1.")
        spectrum["samples"] = int(args.samples)

    if args.pupil_geometry is not None:
        config["pupil"]["geometry"] = args.pupil_geometry
    if args.strehl is not None:
        if not np.isfinite(args.strehl) or not 0.0 <= args.strehl <= 1.0:
            raise ValueError("--strehl must lie in [0, 1].")
        config.setdefault("optics", {})["strehl"] = float(args.strehl)
    if args.phase_mask is not None:
        config["phase_mask"]["name"] = args.phase_mask

    relay = config["relay"]
    if args.edge_angle_deg is not None:
        relay["edge_angle_rad"] = float(np.deg2rad(args.edge_angle_deg))
    if args.edge_offset_mm is not None:
        relay["edge_offset_m"] = float(args.edge_offset_mm) * 1e-3
    if args.cold_stop_x_um is not None:
        relay["cold_stop_x_offset_m"] = float(args.cold_stop_x_um) * 1e-6
    if args.cold_stop_y_um is not None:
        relay["cold_stop_y_offset_m"] = float(args.cold_stop_y_um) * 1e-6

    frame_shape = tuple(args.frame_size)
    if any(size < 1 for size in frame_shape):
        raise ValueError("--frame-size values must be positive integers.")
    height, width = frame_shape

    measurement_mode = bool(args.measure_clear)
    shared_memory_path = None
    fit_summary = None
    clear_background = 0.0
    masked_background = 0.0
    if measurement_mode:
        if args.pupil_center is not None:
            raise ValueError(
                "Do not provide --pupil-center with --measure-clear; the centre "
                "is fitted from the measured clear pupil."
            )
        measured_clear, shared_memory_path = acquire_clear_pupil(
            args.beam_id, args.n_clear, args.frame_sleep
        )
        if measured_clear.shape != frame_shape:
            raise ValueError(
                f"The measured frame has shape {measured_clear.shape}, but "
                f"--frame-size requested {frame_shape}."
            )

        fit_output_directory = args.pupil_fit_output_dir
        if fit_output_directory is None:
            fit_output_directory = args.output.parent / f"{args.output.stem}_pupil_fit"
        fit_summary = fit_pupil_amplitude(
            measured_clear,
            config,
            fit_output_directory,
            write_diagnostics=True,
        )
        fitted_reference_path = fit_output_directory / "updated_references.fits"
        with fits.open(fitted_reference_path) as fitted_hdus:
            predicted_masked = np.asarray(
                fitted_hdus["PHASE_MASK"].data, dtype=float
            ).copy()
            wavelengths = np.linspace(
                float(config["spectrum"]["wavelength_min_m"]),
                float(config["spectrum"]["wavelength_max_m"]),
                int(config["spectrum"]["samples"]),
            )

        clear = measured_clear
        masked = predicted_masked
        clear_background = float(fit_summary["background"])
        masked_background = clear_background
        config = copy.deepcopy(fit_summary["config"])
        centre_result = fit_summary["measured_pupil_center_subpixel"]
        pupil_center_xy = (
            float(centre_result["center_x"]),
            float(centre_result["center_y"]),
        )
    else:
        if args.pupil_center is None:
            pupil_center_xy = ((width - 1.0) / 2.0, (height - 1.0) / 2.0)
        else:
            pupil_center_xy = tuple(args.pupil_center)
        pupil_center_x, pupil_center_y = pupil_center_xy
        if not (0.0 <= pupil_center_x <= width - 1.0):
            raise ValueError("--pupil-center X must lie within the output frame.")
        if not (0.0 <= pupil_center_y <= height - 1.0):
            raise ValueError("--pupil-center Y must lie within the output frame.")

        # Generate the complete binned detector image before the floating-point
        # extraction. The model's ordinary detector.crop uses integer slicing.
        config.setdefault("detector", {})["crop"] = None
        material_file = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "baldr_reference"
            / "Exposed_Ma-N_1405_optical_constants.txt"
        )
        clear_full, masked_full, wavelengths = generate_references(config, material_file)
        clear = extract_frame(clear_full, frame_shape, pupil_center_xy)
        masked = extract_frame(masked_full, frame_shape, pupil_center_xy)

    pupil_center_x, pupil_center_y = pupil_center_xy
    clear, masked, normalization_mask, normalization = normalize_reference_pair(
        clear,
        masked,
        clear_background=clear_background,
        masked_background=masked_background,
    )

    config["output_frame"] = {
        "shape_yx": [height, width],
        "pupil_center_xy_zero_indexed": [pupil_center_x, pupil_center_y],
        "interpolation": "linear",
        "normalization": "mean clear-pupil interior signal",
        "normalization_mask_erosion_pixels": 1,
        "measurement_mode": measurement_mode,
    }
    source_name, _, temperature_k, spectral_type = resolve_source_profile(config)

    common = fits.Header()
    common["BUNIT"] = ("normalized", "Intensity / mean clear-pupil signal")
    common["SOURCE"] = source_name
    common["SRCTEMP"] = (temperature_k, "Resolved blackbody temperature [K]")
    common["SPTYPE"] = (spectral_type or "N/A", "Configured stellar spectral type")
    common["MASKNAME"] = config["phase_mask"].get("name") or "CUSTOM"
    common["NWAVE"] = len(wavelengths)
    common["WAVEMIN"] = (float(wavelengths.min()), "Minimum wavelength [m]")
    common["WAVEMAX"] = (float(wavelengths.max()), "Maximum wavelength [m]")
    common["PUPIL"] = str(config["pupil"].get("geometry") or "CUSTOM").upper()
    common["PUPROT"] = (
        float(config["pupil"].get("rotation_deg", 0.0)),
        "Pupil rotation [deg]",
    )
    common["FRAMENX"] = (width, "Output frame columns")
    common["FRAMENY"] = (height, "Output frame rows")
    common["PUPXCEN"] = (pupil_center_x, "Pupil X centre [zero-indexed pixel]")
    common["PUPYCEN"] = (pupil_center_y, "Pupil Y centre [zero-indexed pixel]")
    common["NORMREF"] = ("CLEAR", "Both frames normalized by clear pupil")
    common["NORMFAC"] = (normalization, "Clear interior mean before normalization")
    common["NORMPIX"] = (
        int(np.count_nonzero(normalization_mask)),
        "Clear interior pixels used for normalization",
    )
    common["CLEARBG"] = (clear_background, "Subtracted clear background")
    common["MASKBG"] = (masked_background, "Subtracted ZWFS background")
    common["MEASMODE"] = (measurement_mode, "Clear pupil measured from shared memory")
    if measurement_mode:
        common["BEAMID"] = (args.beam_id, "Baldr beam ID")
        common["NCLR"] = (args.n_clear, "Averaged clear-pupil frames")
        common["SHMPATH"] = (shared_memory_path, "Camera shared-memory path")
        centre_result = fit_summary["measured_pupil_center_subpixel"]
        if np.isfinite(centre_result["center_sigma_x_pixels"]):
            common["PUPXERR"] = (
                float(centre_result["center_sigma_x_pixels"]),
                "Formal pupil X-centre uncertainty [pixel]",
            )
        if np.isfinite(centre_result["center_sigma_y_pixels"]):
            common["PUPYERR"] = (
                float(centre_result["center_sigma_y_pixels"]),
                "Formal pupil Y-centre uncertainty [pixel]",
            )
        common["PUPCOV"] = (
            float(centre_result["angular_coverage_fraction"]),
            "Outer-edge angular coverage fraction",
        )
    add_complete_config(common, config)

    clear_header = common.copy()
    clear_header["EXTNAME"] = "CLEAR_PUPIL"
    masked_header = common.copy()
    masked_header["EXTNAME"] = "PHASE_MASK"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([
        fits.PrimaryHDU(np.asarray(clear, dtype=np.float64), clear_header),
        fits.ImageHDU(
            np.asarray(masked, dtype=np.float64),
            masked_header,
            name="PHASE_MASK",
        ),
    ]).writeto(args.output, overwrite=args.overwrite)

    print(
        f"Wrote {args.output.resolve()} ({width}x{height} pixels; "
        f"pupil centre x={pupil_center_x:.6f}, y={pupil_center_y:.6f})"
    )
    print(
        f"Normalized both frames by the clear-pupil interior mean "
        f"({normalization:.8g} from {np.count_nonzero(normalization_mask)} pixels)."
    )
    if measurement_mode:
        print(f"Pupil-fit diagnostics: {fit_output_directory.resolve()}")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
