#!/usr/bin/env python3
"""Interactive GUI for Baldr reference generation and alignment recovery."""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.io import fits

# Permit the GUI to sit inside the calibration repository or beside it without
# requiring an editable install. An installed package still takes precedence.
_GUI_DIR = Path(__file__).resolve().parent
_SOURCE_CANDIDATES = (
    _GUI_DIR.parent / "src",
    _GUI_DIR.parent / "baldr_reference_calibration" / "src",
    _GUI_DIR.parent.parent / "baldr_reference_calibration" / "src",
)
for _source_directory in _SOURCE_CANDIDATES:
    if (_source_directory / "baldr_reference" / "__init__.py").is_file():
        sys.path.insert(0, str(_source_directory))
        break

import baldr_reference
from baldr_reference.fitting import (
    PARAMETER_NAMES,
    _registered_extract,
    config_with_parameters,
    fit_alignment,
    measure_pupil,
    parameters_from_config,
)
from baldr_reference.model import generate_references


APP_DIR = _GUI_DIR
MATERIAL = Path(baldr_reference.__file__).with_name("Exposed_Ma-N_1405_optical_constants.txt")
PARAMETER_LABELS = {
    "edge_offset_mm": "Knife-edge offset [mm]",
    "edge_angle_deg": "Knife-edge angle [deg]",
    "cold_stop_x_um": "Cold-stop x [µm]",
    "cold_stop_y_um": "Cold-stop y [µm]",
}


def generate(config):
    working = copy.deepcopy(config)
    working.setdefault("detector", {})["crop"] = None
    return generate_references(working, MATERIAL)[:2]


def fits_bytes(clear, masked):
    buffer = io.BytesIO()
    primary = fits.PrimaryHDU(np.asarray(clear, dtype=np.float64))
    primary.header["EXTNAME"] = "CLEAR_PUPIL"
    fits.HDUList([primary, fits.ImageHDU(masked, name="PHASE_MASK")]).writeto(buffer)
    return buffer.getvalue()


def plot_pair(clear, masked, titles=("Clear pupil", "ZWFS pupil")):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    for axis, image, title in zip(axes, (clear, masked), titles):
        view = axis.imshow(image, origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x [pixel]")
        axis.set_ylabel("y [pixel]")
        fig.colorbar(view, ax=axis, fraction=0.046)
    return fig


def plot_registration(full_clear, measured_clear, fitted_clear, origin):
    from matplotlib.patches import Rectangle

    residual = measured_clear - fitted_clear
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    panels = (
        (full_clear, "Full theoretical frame", "viridis"),
        (measured_clear, "Synthetic measurement", "viridis"),
        (fitted_clear, "Registered model crop", "viridis"),
        (residual, "Residual", "RdBu_r"),
    )
    for axis, (image, title, cmap) in zip(axes, panels):
        kwargs = {}
        if cmap == "RdBu_r":
            limit = np.max(np.abs(image))
            kwargs = {"vmin": -limit, "vmax": limit} if limit else {}
        axis.imshow(image, origin="lower", cmap=cmap, **kwargs)
        axis.set_title(title)
        axis.set_xlabel("x [pixel]")
        axis.set_ylabel("y [pixel]")
    axes[0].add_patch(Rectangle(
        (origin[1] - 0.5, origin[0] - 0.5), measured_clear.shape[1], measured_clear.shape[0],
        fill=False, edgecolor="white", linewidth=2,
    ))
    return fig


def initialise_state():
    if "config_text" not in st.session_state:
        st.session_state.config_text = (APP_DIR / "default_config.json").read_text()
    if "config" not in st.session_state:
        st.session_state.config = json.loads(st.session_state.config_text)
    if "theory" not in st.session_state:
        st.session_state.theory = generate(st.session_state.config)


st.set_page_config(page_title="Baldr reference calibration", page_icon="◉", layout="wide")
initialise_state()

st.title("Baldr reference calibration laboratory")
st.caption("Generate chromatic theoretical references, create realistic detector subframes, and recover relay alignment.")

theory_tab, measurement_tab, fitting_tab = st.tabs([
    "1 · Theory and configuration", "2 · Synthetic measurement", "3 · Alignment fit"
])

with theory_tab:
    editor, display = st.columns([1.05, 1.4], gap="large")
    with editor:
        st.subheader("True configuration")
        edited = st.text_area(
            "Editable JSON", value=st.session_state.config_text, height=650,
            help="Temperature overrides spectral_type when both are supplied.",
        )
        if st.button("Update theoretical references", type="primary", use_container_width=True):
            try:
                candidate = json.loads(edited)
                with st.spinner("Propagating wavelengths through the relay…"):
                    st.session_state.theory = generate(candidate)
                st.session_state.config = candidate
                st.session_state.config_text = json.dumps(candidate, indent=2)
                st.session_state.pop("measurement", None)
                st.session_state.pop("fit", None)
                st.success("Configuration applied.")
            except Exception as error:
                st.error(f"Could not apply configuration: {error}")
    with display:
        clear, masked = st.session_state.theory
        st.subheader("Theoretical detector intensities")
        st.pyplot(plot_pair(clear, masked), clear_figure=True)
        st.write(f"Full detector frame: **{clear.shape[1]} × {clear.shape[0]} pixels**")
        st.download_button(
            "Download theoretical FITS", fits_bytes(clear, masked),
            "theoretical_references.fits", "application/fits", use_container_width=True,
        )

with measurement_tab:
    st.subheader("Measured reference intensities")
    source_mode = st.radio(
        "Measurement source",
        ("Generate synthetic", "Upload measured FITS"),
        horizontal=True,
    )
    nominal = parameters_from_config(st.session_state.config)
    controls, display = st.columns([1, 1.8], gap="large")
    with controls:
        if source_mode == "Generate synthetic":
            crop_size = st.slider("Subframe size [pixel]", 24, 64, 32, 2)
            offset_x = st.slider("Pupil offset x [pixel]", -8.0, 8.0, 3.0, 0.25)
            offset_y = st.slider("Pupil offset y [pixel]", -8.0, 8.0, -2.0, 0.25)
            noise_percent = st.slider("Gaussian noise [% of peak]", 0.0, 10.0, 1.0, 0.1)
            flux_scale = st.slider("Flux scale", 0.1, 3.0, 1.0, 0.05)
            background_percent = st.slider("Background [% of peak]", 0.0, 20.0, 1.0, 0.25)
            seed = st.number_input("Random seed", min_value=0, value=5, step=1)
            st.markdown("**True optical alignment**")
            truth = np.array([
                st.slider("Knife-edge offset [mm]", -1.30, -0.50, float(nominal[0]), 0.01),
                st.slider("Knife-edge angle [deg]", -4.0, 4.0, float(nominal[1]), 0.1),
                st.slider("Cold-stop x [µm]", -120.0, 120.0, float(nominal[2]), 5.0),
                st.slider("Cold-stop y [µm]", -120.0, 120.0, float(nominal[3]), 5.0),
            ])
            make_measurement = st.button(
                "Generate measured intensities", type="primary", use_container_width=True
            )
            if make_measurement:
                with st.spinner("Generating and sampling the synthetic measurement…"):
                    truth_config = config_with_parameters(st.session_state.config, truth)
                    full_clear, full_masked = generate(truth_config)
                    full_info = measure_pupil(full_clear)
                    full_center = (full_info["center_y"], full_info["center_x"])
                    shape = (crop_size, crop_size)
                    measured_center = (
                        (crop_size - 1) / 2 + offset_y, (crop_size - 1) / 2 + offset_x
                    )
                    clear = _registered_extract(full_clear, shape, full_center, measured_center)
                    masked = _registered_extract(full_masked, shape, full_center, measured_center)
                    rng = np.random.default_rng(int(seed))
                    clear_peak, masked_peak = clear.max(), masked.max()
                    clear = flux_scale * clear + background_percent / 100 * clear_peak
                    masked = flux_scale * masked + background_percent / 100 * masked_peak
                    clear += rng.normal(0, noise_percent / 100 * clear_peak, shape)
                    masked += rng.normal(0, noise_percent / 100 * masked_peak, shape)
                    st.session_state.measurement = {
                        "source": "synthetic", "clear": clear, "masked": masked,
                        "truth": truth, "full_clear": full_clear, "full_masked": full_masked,
                        "true_origin": (
                            full_center[0] - measured_center[0],
                            full_center[1] - measured_center[1],
                        ),
                    }
                    st.session_state.pop("fit", None)
        else:
            uploaded = st.file_uploader(
                "Clear + ZWFS reference FITS",
                type=("fits", "fit", "fts"),
                help=(
                    "Accepts named CLEAR_PUPIL/PHASE_MASK image extensions, "
                    "the first two 2-D HDUs, or the first two planes of a cube."
                ),
            )
            if st.button(
                "Load measured FITS", type="primary", use_container_width=True,
                disabled=uploaded is None,
            ):
                try:
                    clear, masked = read_reference_fits(io.BytesIO(uploaded.getvalue()))
                    full_clear, full_masked = st.session_state.theory
                    st.session_state.measurement = {
                        "source": "uploaded", "filename": uploaded.name,
                        "clear": clear, "masked": masked, "truth": None,
                        "full_clear": full_clear, "full_masked": full_masked,
                        "true_origin": None,
                    }
                    st.session_state.pop("fit", None)
                    st.success(f"Loaded {uploaded.name}: {clear.shape[1]} × {clear.shape[0]} pixels.")
                except Exception as error:
                    st.error(f"Could not read the uploaded FITS: {error}")

    with display:
        if "measurement" not in st.session_state:
            st.info("Choose the detector and alignment settings, then generate a measurement.")
        else:
            measurement = st.session_state.measurement
            label = "Uploaded" if measurement.get("source") == "uploaded" else "Synthetic"
            st.caption(f"{label} clear and ZWFS reference pair")
            st.pyplot(plot_pair(
                measurement["clear"], measurement["masked"],
                ("Measured clear pupil", "Measured ZWFS pupil"),
            ), clear_figure=True)
            st.download_button(
                "Download measured FITS",
                fits_bytes(measurement["clear"], measurement["masked"]),
                "measured_references.fits", "application/fits", use_container_width=True,
            )

with fitting_tab:
    if "measurement" not in st.session_state:
        st.warning("Generate or upload a measurement in tab 2 first.")
    else:
        st.subheader("Recover knife-edge and cold-stop alignment")
        if st.button("Run coarse search and fine fit", type="primary"):
            measurement = st.session_state.measurement
            with st.spinner("Fitting alignment; this can take tens of seconds…"):
                with tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary)
                    summary = fit_alignment(
                        measurement["clear"], measurement["masked"],
                        st.session_state.config, output,
                    )
                    products = fits.open(output / "fit_products.fits")
                    fitted_clear = np.asarray(products[2].data).copy()
                    fitted_masked = np.asarray(products[3].data).copy()
                    products.close()
                    st.session_state.fit = {
                        "summary": summary,
                        "fitted_clear": fitted_clear,
                        "fitted_masked": fitted_masked,
                        "maps_png": (output / "coarse_fit_maps.png").read_bytes(),
                        "products_fits": (output / "fit_products.fits").read_bytes(),
                    }

        if "fit" in st.session_state:
            fit = st.session_state.fit
            measurement = st.session_state.measurement
            summary = fit["summary"]
            fitted = np.asarray(summary["fitted_parameters"])
            rows = []
            if measurement.get("truth") is not None:
                truth = np.asarray(measurement["truth"])
                for name, true_value, fitted_value in zip(PARAMETER_NAMES, truth, fitted):
                    rows.append({
                        "Parameter": PARAMETER_LABELS[name], "True": true_value,
                        "Fitted": fitted_value, "Fitted − true": fitted_value - true_value,
                    })
            else:
                initial = np.asarray(summary["initial_parameters"])
                for name, initial_value, fitted_value in zip(PARAMETER_NAMES, initial, fitted):
                    rows.append({
                        "Parameter": PARAMETER_LABELS[name], "Initial": initial_value,
                        "Fitted": fitted_value, "Fitted − initial": fitted_value - initial_value,
                    })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.pyplot(plot_registration(
                measurement["full_clear"], measurement["clear"], fit["fitted_clear"],
                summary["model_crop_origin_yx"],
            ), clear_figure=True)
            st.image(fit["maps_png"], caption="Coarse grid-search maps")
            left, right = st.columns(2)
            left.metric("Recovered crop y", f"{summary['model_crop_origin_yx'][0]:.3f} px")
            right.metric("Recovered crop x", f"{summary['model_crop_origin_yx'][1]:.3f} px")
            st.download_button(
                "Download fit products FITS", fit["products_fits"],
                "fit_products.fits", "application/fits",
            )
            st.download_button(
                "Download fit summary JSON", json.dumps(summary, indent=2),
                "fit_summary.json", "application/json",
            )
