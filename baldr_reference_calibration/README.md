# Baldr reference calibration

Standalone tools for:

1. Generating theoretical clear pupil and ZWFS reference intensities.
2. Fitting knife edge and cold stop alignment from a measured clear/ZWFS pair.
3. Testing alignment recovery with synthetic data.
4. Measuring sensitivity to noise, phase mask , source, and passband errors.

The package does not import BaldrApp, pyZELDA, or XAOSIM. It preserves the
validated BaldrApp 0.1.7 numerical conventions and requires Python 3.11+.

The physical model integrates a configurable internal or stellar spectrum over wavelength and propagates it through the Baldr system to generate clear pupil and ZWFS intensities. It includes configurable pupil geometry and rotation, flat-DM phase, wavelength dependent phase mask  size and phase delay, Fresnel relay propagation, knife edge and cold stop alignment, pupil misconjugation, detector sampling, and photometric scaling. The fitting routine automatically registers and crops the theoretical pupil to the measured subframe before fitting the knife edge and cold stop parameters.

## Installation

From this directory, create or activate an isolated environment and install:

```bash
python -m pip install -e ".[test]"
```

This provides both Python modules and the following commands:

```text
baldr-reference-generate
baldr-reference-fit
baldr-reference-sim-test
baldr-reference-sensitivity
baldr-reference-compare
```

Thin wrappers with equivalent behavior are retained under `scripts/`.

## Generate a theoretical reference

```bash
baldr-reference-generate \
  configs/reference_internal.example.json \
  output/reference_intensities.fits
```

The primary image is `CLEAR_PUPIL`; extension 1 is `PHASE_MASK`. Set
`detector.crop` to `null` for the complete binned image or `[32, 32]` for a
centered crop. Both headers store a lossless copy of the input configuration.

Use `configs/reference_onsky.example.json` for the example on-sky source.
The spectral wavelength limits define the propagated passband. The phase mask 
beam `optics.f_number` remains required for converting physical mask diameters
to wavelength dependent lambda/D units; no separate reference wavelength is
required.

## Interactive GUI

The optional Streamlit application is contained under `gui/`. Install its UI
dependency into the same environment and launch it from the repository root:

```bash
python -m pip install -r gui/requirements.txt
python -m streamlit run gui/app.py
```

Its three tabs provide an editable true configuration and theoretical pupil
plots, either tunable noisy/cropped measurements or an uploaded measured FITS
pair, and alignment fitting with crop,
residual, grid-search, and parameter-recovery diagnostics. The GUI can export
the theoretical, measured, and fitted products as FITS and JSON files.

## Fit measured references

```bash
baldr-reference-fit \
  measured_references.fits \
  configs/fit.example.json \
  fit_output
```

The FITS reader accepts named clear/ZWFS extensions, the first two 2-D image
HDUs, or the first two frames of a cube. It produces a summary, coarse fit
maps, fitted images, residuals, and the fit mask.

By default, `fit.registration` is `"auto"`. The fitter measures the geometric
pupil centre from the clear image, then extracts the matching region from the
larger theoretical detector image. This supports off-centre subframes such as
32 x 32 measurements while retaining a 64 x 64 theoretical model. The same
fixed registration is applied to the clear and ZWFS models throughout the
optical fit. Use `"frame_center"` only when both frames are already centred.

Visualize registration with a synthetic off-centre 32 x 32 subframe:

```bash
baldr-reference-registration \
  configs/fit.example.json \
  output/registration_test \
  --crop-size 32 --offset-x 3 --offset-y -2
```

To inspect registration of a real FITS pair instead:

```bash
baldr-reference-registration \
  configs/fit.example.json \
  output/registration_measured \
  --references measured_references.fits
```

The output includes `registration_and_crop.png`, a multi-extension diagnostic
FITS file, and a JSON report containing the detected centres and crop origin.

## Test before deployment

Run the fast unit tests:

```bash
pytest -m "not slow"
```

Run the BaldrApp regression and off-grid synthetic recovery tests:

```bash
pytest -m slow
```

Run everything:

```bash
pytest
```

Exercise the operator commands directly:

```bash
baldr-reference-generate \
  configs/reference_internal.example.json \
  output/reference_intensities.fits

baldr-reference-compare \
  tests/data/baldrapp_reference_intensities.fits \
  output/reference_intensities.fits

baldr-reference-sim-test \
  configs/synthetic_recovery.example.json \
  synthetic_test_output
```

The regression comparison should report relative L2 errors near `1e-15` and
flux ratios near `1.0`. The synthetic recovery report should stay within the
tolerances enforced by `tests/test_synthetic_recovery.py`.

The full sensitivity analysis is intentionally separate from routine tests:

```bash
baldr-reference-sensitivity \
  configs/sensitivity.example.json \
  sensitivity_output
```

It produces per-run CSV data, grouped JSON statistics, and a four-panel error
plot. The checked-in `examples/sensitivity_baseline/` shows one validated run.

## Repository layout

```text
src/baldr_reference/       reusable implementation
scripts/                   direct operator wrappers
configs/                   operational and test examples
data/                      source copy of material optical constants
tests/                     unit and integration tests
tests/data/                fixed BaldrApp regression reference
examples/                  documented generated examples
gui/                       optional interactive Streamlit application
```

Generated output directories are excluded by `.gitignore`.
