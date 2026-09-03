from pathlib import Path

import numpy as np
import pytest

from baldr_reference.fitting import _registered_extract, config_with_parameters, fit_alignment, measure_pupil
from baldr_reference.model import generate_references, load_config


@pytest.mark.slow
def test_off_grid_alignment_recovery(tmp_path):
    config = load_config("configs/synthetic_recovery.example.json")
    truth = np.asarray(config["synthetic_test"]["true_parameters"], dtype=float)
    material = Path("src/baldr_reference/Exposed_Ma-N_1405_optical_constants.txt")
    clear, masked, _ = generate_references(config_with_parameters(config, truth), material)
    result = fit_alignment(clear, masked, config, tmp_path, write_diagnostics=False)
    fitted = np.asarray(result["fitted_parameters"])
    assert np.all(np.abs(fitted - truth) < np.array([0.005, 0.1, 5.0, 5.0]))


@pytest.mark.slow
def test_off_center_32_pixel_subframe_recovery(tmp_path):
    config = load_config("configs/synthetic_recovery.example.json")
    config["fit"]["registration"] = "auto"
    truth = np.asarray(config["synthetic_test"]["true_parameters"], dtype=float)
    material = Path("src/baldr_reference/Exposed_Ma-N_1405_optical_constants.txt")
    clear, masked, _ = generate_references(config_with_parameters(config, truth), material)
    center = measure_pupil(clear)
    model_center = (center["center_y"], center["center_x"])
    measured_center = (13.5, 18.5)
    clear = _registered_extract(clear, (32, 32), model_center, measured_center)
    masked = _registered_extract(masked, (32, 32), model_center, measured_center)
    result = fit_alignment(clear, masked, config, tmp_path, write_diagnostics=False)
    fitted = np.asarray(result["fitted_parameters"])
    assert np.all(np.abs(fitted - truth) < np.array([0.010, 0.15, 8.0, 8.0]))
    assert np.allclose(result["measured_pupil_center_yx"], measured_center, atol=0.6)
