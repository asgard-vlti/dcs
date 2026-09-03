#!/usr/bin/env python3
"""Generate standalone theoretical Baldr reference intensities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from .model import generate_references, load_config, resolve_source_profile


def add_complete_config(header: fits.Header, config: dict) -> None:
    """Store a losslessly reconstructable copy of the complete JSON config."""
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    chunks = [serialized[index:index + 60] for index in range(0, len(serialized), 60)]
    header["CFGNCHNK"] = (len(chunks), "Number of ordered JSON config chunks")
    for index, chunk in enumerate(chunks):
        header[f"CFG{index:04d}"] = chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    source_name, _, temperature_k, spectral_type = resolve_source_profile(config)
    material_file = Path(__file__).with_name("Exposed_Ma-N_1405_optical_constants.txt")
    clear, masked, wavelengths = generate_references(config, material_file)

    common = fits.Header()
    common["BUNIT"] = ("electron", "Expected photoelectrons per exposure")
    common["SOURCE"] = source_name
    common["SRCTEMP"] = (temperature_k, "Resolved blackbody temperature [K]")
    common["SPTYPE"] = (spectral_type or "N/A", "Configured stellar spectral type")
    common["MASKNAME"] = config["phase_mask"].get("name") or "CUSTOM"
    common["NWAVE"] = len(wavelengths)
    common["WAVEMIN"] = (float(wavelengths.min()), "Minimum wavelength [m]")
    common["WAVEMAX"] = (float(wavelengths.max()), "Maximum wavelength [m]")
    common["PUPIL"] = str(config["pupil"].get("geometry") or "CUSTOM").upper()
    common["PUPROT"] = (float(config["pupil"].get("rotation_deg", 0.0)), "Pupil rotation [deg]")
    add_complete_config(common, config)

    clear_header = common.copy()
    clear_header["EXTNAME"] = "CLEAR_PUPIL"
    masked_header = common.copy()
    masked_header["EXTNAME"] = "PHASE_MASK"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([
        fits.PrimaryHDU(np.asarray(clear, dtype=np.float64), clear_header),
        fits.ImageHDU(np.asarray(masked, dtype=np.float64), masked_header, name="PHASE_MASK"),
    ]).writeto(args.output, overwrite=args.overwrite)
    print(f"Wrote {args.output.resolve()} ({clear.shape[1]}x{clear.shape[0]} pixels)")


if __name__ == "__main__":
    main()
