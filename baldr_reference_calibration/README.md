# Baldr reference calibration

Standalone tools for:

1. Acquiring averaged clear pupil and ZWFS reference intensities from shared memory.
2. Generating theoretical clear pupil and ZWFS reference intensities.
3. Fitting knife edge and cold stop alignment from a measured clear/ZWFS pair.
4. Testing alignment recovery with synthetic data.
5. Measuring sensitivity to noise, phase mask, source, and passband errors.
6. Fitting an on-sky pupil amplitude and generating an updated ZWFS reference.

The physical model and fitting routines do not import BaldrApp, pyZELDA, or
XAOSIM. The acquisition command uses XAOSIM only to access the camera shared
memory. The model preserves the validated BaldrApp 0.1.7 numerical conventions
and requires Python 3.11+.

The physical model integrates a configurable internal or stellar spectrum over wavelength and propagates it through the Baldr system to generate clear pupil and ZWFS intensities. It includes configurable pupil geometry and rotation, flat-DM phase, wavelength dependent phase mask  size and phase delay, Fresnel relay propagation, knife edge and cold stop alignment, pupil misconjugation, detector sampling, and photometric scaling. The fitting routine automatically registers and crops the theoretical pupil to the measured subframe before fitting the knife edge and cold stop parameters.

The top-level operational workflow is:

1. Acquire averaged clear-pupil (`N0`) and ZWFS-pupil (`I0`) images from shared memory.
2. Fit the measured pair using a JSON configuration appropriate to the beam, phase mask, source, and passband.
3. Inspect the fitted parameters, residuals, and fit diagnostics. If the inferred alignment is unacceptable, adjust the optics and repeat the acquisition and fit. If it is acceptable, transfer the fitted alignment values into the relevant configuration and generate or update the reference products used for subsequent control-loop calibration.
4. For on-sky operation, acquire a clear pupil, fit its geometry and smooth amplitude with the internal relay alignment held fixed, and generate the corresponding on-sky ZWFS reference.

Acquisition and fitting are diagnostic steps: they do not move the phase mask, adjust the optics, or update an RTC configuration automatically.

## Installation

From this directory, create or activate an isolated environment and install:

```bash
python -m pip install -e ".[test]"
```

This provides both Python modules and the following commands:

```text
baldr-reference-generate
baldr-reference-fit
baldr-acquire-references
baldr-reference-sim-test
baldr-reference-sensitivity
baldr-reference-compare
baldr-reference-pupil-fit
```

Thin wrappers with equivalent behavior are retained under `scripts/`.

## Generate reference intensities with `gen_refs.py`

`scripts/gen_refs.py` is the main reference-generation workflow. It reads a
complete JSON configuration, while source, wavelength, pupil, Strehl, phase
mask, relay-alignment, frame-size, and pupil-centre values can be overridden
with command-line arguments. Only the output path is required.

### Mode 1: purely theoretical references

Generate both references from the configured model:

```bash
python scripts/gen_refs.py output/reference_H3.fits \
  --config configs/reference_onsky.example.json \
  --phase-mask H3 \
  --spectral-type K5V \
  --pupil-geometry at \
  --strehl 0.8 \
  --frame-size 32 32 \
  --pupil-center 15.25 15.75 \
  --overwrite
```

`--pupil-center` is `(x, y)` in zero-indexed floating-point detector pixels.

### Mode 2: measured clear pupil and modelled ZWFS reference

```bash
python scripts/gen_refs.py output/reference_H3.fits \
  --config configs/reference_onsky.example.json \
  --measure-clear \
  --beam-id 1 \
  --n-clear 500 \
  --pupil-geometry at \
  --phase-mask H3 \
  --frame-size 32 32 \
  --overwrite
```

This mode prompts the operator to move the phase mask out, measures the clear
pupil from shared memory, and fits its subpixel centre, scale, rotation, and
smooth amplitude. Internal references use the Solarstein pupil; on sky, the AT
or UT prior supplies the appropriate central obstruction and spider geometry.
The overall pupil rotation and registration are then fitted from the measured
clear pupil before it is propagated through the ZWFS and fixed relay model.

In both modes, `CLEAR_PUPIL` is the primary image and `PHASE_MASK` is extension
1. Both are divided by the same mean clear-pupil interior signal; the ZWFS
reference is not normalized independently. The complete effective configuration
is stored in the FITS headers, and measured mode also saves pupil-fit and
residual diagnostics.

### Simplified theoretical generator

`gen_refs.py` evolved from the configuration-only generator, which remains
available for simple full-frame or centred theoretical references:

```bash
baldr-reference-generate \
  configs/reference_internal.example.json \
  output/reference_intensities.fits
```

Set `detector.crop` to `null` for the complete binned image or `[32, 32]` for a
centred crop.

## Acquire measured clear and ZWFS references

To acquire both references directly from shared memory instead of modelling
the ZWFS reference, run:

```bash
baldr-acquire-references \
  --beam_id 1 \
  --n_clear 500 \
  --n_zwfs 500 \
  --phasemask H3 \
  --output_dir output
```

The operator is prompted to move the phase mask out for `N0` and then into the
beam for `I0`. The command does not move any hardware itself.

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

The JSON input is required because the two measured pupil images do not by
themselves define the complete physical forward model. It supplies the nominal
instrument and observing configuration, including the wavelength sampling and
source spectrum, pupil and detector geometry, phase-mask properties, relay
parameters, and initial alignment values. It also defines the fitting settings,
including parameter bounds, finite-difference steps, coarse-search sampling,
and convergence limits. The alignment values in the JSON are initial conditions
rather than assumed true values: the knife-edge offset and angle and the two
cold-stop offsets are varied within the configured bounds during the fit.

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

## Fit an on-sky pupil and update the ZWFS reference

After fitting the stable relay alignment from internal clear/ZWFS references,
copy those fitted relay values into an on-sky configuration and run:

```bash
baldr-reference-pupil-fit \
  measured_onsky_references.fits \
  configs/pupil_fit.example.json \
  output/onsky_pupil_fit
```

Only the measured clear pupil is used; the input FITS may contain either one
clear image or the usual clear/ZWFS pair. The relay alignment remains fixed.
The configured AT, UT, Solarstein, disc, or custom pupil supplies the geometric
prior. The fit first performs a coarse rotation search and then jointly refines
absolute rotation, high-resolution pupil translation, and diameter scale. It
can subsequently fit a regularized coarse illumination map. No pupil phase is
inferred from the intensity measurement.

The model sampling is set by `pupil.array_size` and
`pupil.pixels_across_pupil`; it may be finer than the detector sampling because
the forward model applies the configured detector binning after propagation.
The smooth amplitude grid is a regularized description of illumination, not an
attempt to recover unconstrained structure below the detector resolution.

`pupil_fit.example.json` documents the bounds and regularization settings. For
a full detector frame, pupil translation is fitted in the model coordinates.
For a cropped frame whose full-frame origin is unavailable, pupil centres are
registered first with a robust subpixel fit to the outer pupil edge. The fit
includes a planar illumination term so flux gradients do not bias the centre;
spiders and local defects are rejected by the robust loss. Translation remains
partly degenerate with the unknown crop origin, while rotation, scale, and
illumination are constrained by morphology.

The output directory contains:

```text
pupil_fit_summary.json
fitted_config.json
pupil_fit_products.fits
updated_references.fits
pupil_fit_diagnostics.png
pupil_rotation_scan.png
pupil_center_fit.png
```

`updated_references.fits` contains `CLEAR_PUPIL` and `PHASE_MASK` in the same
layout as `baldr-reference-generate`. The ZWFS image is generated by inserting
the fitted high-resolution amplitude before the phase-mask propagation, then
applying the fixed relay and `optics.strehl`.

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

Run the complete internal-calibration to on-sky-reference stress test with:

```bash
python bens_playground/stress_test_reference_pipeline.py
```

This uses `configs/stress_internal.example.json` for a 1900 K Solarstein
source and `configs/stress_onsky.example.json` for a K5V AT pupil whose true
rotation is 30 degrees. It injects relay misalignment, detector noise, pupil
translation, a diameter error, and a smooth amplitude gradient; fits the
internal alignment; transfers that alignment to the on-sky configuration; and
tests the resulting updated ZWFS reference against the known synthetic truth.

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
