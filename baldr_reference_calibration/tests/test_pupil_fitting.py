from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from baldr_reference.model import generate_clear_reference, load_config, make_pupil
from baldr_reference.pupil_fitting import (
    correction_from_grid,
    fit_pupil_amplitude,
    read_clear_fits,
    transform_pupil,
)


def test_identity_pupil_transform():
    pupil = np.zeros((24, 24))
    pupil[5:19, 6:18] = 1.0
    transformed = transform_pupil(pupil, 0.0, 0.0, 0.0, 1.0)
    assert np.array_equal(transformed, pupil)


def test_coarse_correction_interpolates_to_pupil_grid():
    coefficients = np.array([[-0.1, 0.1], [-0.1, 0.1]])
    correction = correction_from_grid(coefficients, (20, 30))
    assert correction.shape == (20, 30)
    assert np.allclose(correction[:, 0], -0.1)
    assert np.allclose(correction[:, -1], 0.1)


def test_clear_reader_accepts_single_image(tmp_path):
    image = np.arange(36, dtype=float).reshape(6, 6)
    path = tmp_path / "clear.fits"
    fits.PrimaryHDU(image).writeto(path)
    assert np.array_equal(read_clear_fits(path), image)


@pytest.mark.slow
def test_geometric_pupil_recovery(tmp_path):
    config = load_config("configs/reference_onsky.example.json")
    config["pupil"].update(
        geometry="AT", array_size=96, pixels_across_pupil=32, rotation_deg=0.0
    )
    config["detector"].update(binning=3, crop=None)
    config["spectrum"]["samples"] = 1
    config["pupil_fit"] = {
        "rotation_bounds_deg": [-12, 12],
        "rotation_coarse_points": 7,
        "shift_x_bounds_pixels": [-2, 2],
        "shift_y_bounds_pixels": [-2, 2],
        "diameter_scale_bounds": [0.97, 1.04],
        "geometric_max_function_evaluations": 30,
        "fit_smooth_amplitude": False,
    }
    truth = np.array([7.0, 0.8, -0.6, 1.015])
    amplitude = transform_pupil(make_pupil(config), *truth)
    measured_clear, _ = generate_clear_reference(config, amplitude)
    result = fit_pupil_amplitude(
        measured_clear, config, tmp_path, write_diagnostics=False
    )
    fitted = np.asarray(result["fitted_parameters"])
    assert np.allclose(fitted, truth, atol=[0.02, 0.02, 0.02, 2e-4])
    assert (tmp_path / "updated_references.fits").is_file()
    assert (tmp_path / "pupil_fit_products.fits").is_file()
