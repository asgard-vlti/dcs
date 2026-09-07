#!/usr/bin/env python3
"""Compare a standalone FITS product with a BaldrApp reference product."""

import argparse
from astropy.io import fits
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    with fits.open(args.reference) as expected, fits.open(args.candidate) as actual:
        for index, name in ((0, "CLEAR_PUPIL"), (1, "PHASE_MASK")):
            a = np.asarray(expected[index].data, dtype=float)
            b = np.asarray(actual[index].data, dtype=float)
            difference = b - a
            relative_l2 = np.linalg.norm(difference) / np.linalg.norm(a)
            print(
                f"{name}: shape={b.shape}, relative_L2={relative_l2:.6e}, "
                f"max_abs={np.max(np.abs(difference)):.6e}, "
                f"flux_ratio={b.sum() / a.sum():.12f}"
            )


if __name__ == "__main__":
    main()
