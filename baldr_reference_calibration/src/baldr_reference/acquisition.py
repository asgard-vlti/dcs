#!/usr/bin/env python3
"""
Acquire averaged Baldr clear-pupil (N0) and ZWFS-pupil (I0) references
directly from shared memory and save them in the FITS convention expected
by the DCS fitting routine.

The user is responsible for moving the phase mask:

    1. move the phase mask OUT, then press Enter to acquire N0;
    2. move the phase mask IN, then press Enter to acquire I0.

No motor commands are sent by this script.

FITS layout
-----------
The DCS fitter explicitly recognises the extension names:

    N0   -> clear pupil
    I0   -> ZWFS / phase-mask pupil

The clear image is written first and the masked image second so that the
file is also correct if the fitter falls back to image order.

Example
-------
python acquire_baldr_reference_pupils_for_fitting.py \
    --beam_id 1 \
    --n_clear 500 \
    --n_zwfs 500 \
    --phasemask H3
"""

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from xaosim.shmlib import shm


# ============================================================
# Arguments
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Acquire averaged Baldr clear and ZWFS pupils from shared memory "
            "in the FITS convention expected by the DCS fitting routine."
        )
    )

    parser.add_argument(
        "--beam_id",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Baldr beam ID. Default: 1",
    )

    parser.add_argument(
        "--n_clear",
        type=int,
        required=True,
        help="Number of frames to average for the clear pupil N0.",
    )

    parser.add_argument(
        "--n_zwfs",
        type=int,
        required=True,
        help="Number of frames to average for the ZWFS pupil I0.",
    )

    parser.add_argument(
        "--frame_sleep",
        type=float,
        default=0.01,
        help="Sleep between shared-memory reads in seconds. Default: 0.01",
    )

    parser.add_argument(
        "--phasemask",
        type=str,
        default="UNKNOWN",
        help="Phase-mask label written to FITS metadata only, e.g. H3.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Output directory. Default: current directory.",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional output FITS filename.",
    )

    args = parser.parse_args()


    # ============================================================
    # Checks
    # ============================================================

    if args.n_clear <= 0:
        raise ValueError("--n_clear must be > 0")

    if args.n_zwfs <= 0:
        raise ValueError("--n_zwfs must be > 0")

    if args.frame_sleep < 0:
        raise ValueError("--frame_sleep must be >= 0")


    # ============================================================
    # Open camera shared memory
    # ============================================================

    shm_path = f"/dev/shm/baldr{args.beam_id}.im.shm"

    print()
    print(f"Opening shared memory: {shm_path}")

    cam_shm = shm(shm_path)

    test_frame = np.asarray(cam_shm.get_data())

    if test_frame.ndim != 2:
        raise RuntimeError(
            f"Expected a 2-D Baldr subframe, got shape {test_frame.shape}"
        )

    print(f"Image shape: {test_frame.shape}")
    print(f"Input dtype: {test_frame.dtype}")


    # ============================================================
    # Acquire clear pupil N0
    # ============================================================

    print()
    print("============================================================")
    print("CLEAR PUPIL N0")
    print("Move the phase mask OUT of the beam.")
    print("Make sure the pupil is stable.")
    input("Press Enter to start acquiring N0... ")

    clear_frames = []

    for frame_index in range(args.n_clear):

        frame = np.asarray(
            cam_shm.get_data(),
            dtype=np.float64,
        )

        if frame.shape != test_frame.shape:
            raise RuntimeError(
                f"Camera frame shape changed from {test_frame.shape} "
                f"to {frame.shape}"
            )

        clear_frames.append(frame)

        if args.frame_sleep > 0:
            time.sleep(args.frame_sleep)

        if (
            frame_index == 0
            or (frame_index + 1) % max(1, args.n_clear // 10) == 0
            or frame_index + 1 == args.n_clear
        ):
            print(
                f"  N0: {frame_index + 1}/{args.n_clear}"
            )

    N0 = np.mean(
        np.asarray(clear_frames),
        axis=0,
    ).astype(np.float32)

    print()
    print("N0 acquisition complete.")
    print(f"Mean: {np.mean(N0):.3f}")
    print(f"Peak: {np.max(N0):.3f}")


    # ============================================================
    # Acquire ZWFS pupil I0
    # ============================================================

    print()
    print("============================================================")
    print("ZWFS PUPIL I0")
    print(
        f"Move phase mask {args.phasemask} INTO the beam and align it."
    )
    print("Make sure the pupil is stable.")
    input("Press Enter to start acquiring I0... ")

    zwfs_frames = []

    for frame_index in range(args.n_zwfs):

        frame = np.asarray(
            cam_shm.get_data(),
            dtype=np.float64,
        )

        if frame.shape != test_frame.shape:
            raise RuntimeError(
                f"Camera frame shape changed from {test_frame.shape} "
                f"to {frame.shape}"
            )

        zwfs_frames.append(frame)

        if args.frame_sleep > 0:
            time.sleep(args.frame_sleep)

        if (
            frame_index == 0
            or (frame_index + 1) % max(1, args.n_zwfs // 10) == 0
            or frame_index + 1 == args.n_zwfs
        ):
            print(
                f"  I0: {frame_index + 1}/{args.n_zwfs}"
            )

    I0 = np.mean(
        np.asarray(zwfs_frames),
        axis=0,
    ).astype(np.float32)

    print()
    print("I0 acquisition complete.")
    print(f"Mean: {np.mean(I0):.3f}")
    print(f"Peak: {np.max(I0):.3f}")


    # ============================================================
    # Output path
    # ============================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S"
    )

    if args.filename is None:
        output_path = (
            args.output_dir
            / (
                f"baldr_reference_fit_"
                f"B{args.beam_id}_"
                f"{args.phasemask}_"
                f"{timestamp}.fits"
            )
        )
    else:
        output_path = args.output_dir / args.filename

    if output_path.suffix.lower() not in {
        ".fits",
        ".fit",
        ".fts",
    }:
        output_path = output_path.with_suffix(".fits")


    # ============================================================
    # Write FITS in the DCS fitting convention
    # ============================================================

    primary = fits.PrimaryHDU()

    primary.header["BEAMID"] = (
        args.beam_id,
        "Baldr beam ID",
    )

    primary.header["PHMASK"] = (
        args.phasemask,
        "Phase mask used for I0",
    )

    primary.header["NCLR"] = (
        args.n_clear,
        "Frames averaged for N0",
    )

    primary.header["NZWFS"] = (
        args.n_zwfs,
        "Frames averaged for I0",
    )

    primary.header["SHMPATH"] = (
        shm_path,
        "Camera shared-memory path",
    )

    primary.header["DATE"] = (
        datetime.datetime.now().isoformat(
            timespec="seconds"
        ),
        "File creation time",
    )

    n0_hdu = fits.ImageHDU(
        np.asarray(N0, dtype=np.float32),
        name="N0",
    )

    n0_hdu.header["NAVG"] = (
        args.n_clear,
        "Number of averaged frames",
    )

    n0_hdu.header["MASKPOS"] = (
        "OUT",
        "Phase-mask state",
    )

    i0_hdu = fits.ImageHDU(
        np.asarray(I0, dtype=np.float32),
        name="I0",
    )

    i0_hdu.header["NAVG"] = (
        args.n_zwfs,
        "Number of averaged frames",
    )

    i0_hdu.header["PHMASK"] = (
        args.phasemask,
        "Phase mask",
    )

    fits.HDUList(
        [
            primary,
            n0_hdu,
            i0_hdu,
        ]
    ).writeto(
        output_path,
        overwrite=True,
    )


    # ============================================================
    # Verify using the same logic expected by fitting.py
    # ============================================================

    with fits.open(output_path) as hdus:

        names = [
            str(
                hdu.header.get(
                    "EXTNAME",
                    hdu.name,
                )
            ).strip().upper()
            for hdu in hdus
            if hdu.data is not None
        ]

        if "N0" not in names:
            raise RuntimeError(
                "Written FITS does not contain an N0 extension."
            )

        if "I0" not in names:
            raise RuntimeError(
                "Written FITS does not contain an I0 extension."
            )

        N0_check = np.asarray(
            hdus["N0"].data,
            dtype=float,
        )

        I0_check = np.asarray(
            hdus["I0"].data,
            dtype=float,
        )

    if N0_check.shape != I0_check.shape:
        raise RuntimeError(
            "Written N0 and I0 shapes differ."
        )

    if not np.all(np.isfinite(N0_check)):
        raise RuntimeError(
            "Written N0 contains non-finite values."
        )

    if not np.all(np.isfinite(I0_check)):
        raise RuntimeError(
            "Written I0 contains non-finite values."
        )


    # ============================================================
    # Summary
    # ============================================================

    print()
    print("============================================================")
    print("Saved reference FITS for DCS fitting")
    print(f"File: {output_path.resolve()}")
    print()
    print("FITS layout:")
    print("  PRIMARY : metadata only")
    print(
        f"  N0      : {N0.shape}, float32, "
        f"mean of {args.n_clear} clear frames"
    )
    print(
        f"  I0      : {I0.shape}, float32, "
        f"mean of {args.n_zwfs} ZWFS frames"
    )
    print()
    print(
        "This layout is explicitly recognised by "
        "fitting.read_reference_fits()."
    )



if __name__ == "__main__":
    main()