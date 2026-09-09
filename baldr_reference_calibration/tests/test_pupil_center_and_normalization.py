from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from baldr_reference.pupil_fitting import estimate_pupil_center_subpixel


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_reference_fits_with_overrides.py"
)
SPEC = importlib.util.spec_from_file_location("generate_reference_overrides", SCRIPT_PATH)
WRAPPER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(WRAPPER)


def _synthetic_at_pupil(center_x, center_y, seed):
    rng = np.random.default_rng(seed)
    y, x = np.indices((32, 32), dtype=float)
    radius = np.hypot(x - center_x, y - center_y)
    outer = 1.0 / (1.0 + np.exp((radius - 7.7) / 0.3))
    inner = 1.0 / (1.0 + np.exp(-(radius - 2.0) / 0.3))
    image = 800.0 * outer * inner * (
        1.0
        + 0.18 * (x - center_x) / 7.7
        + 0.10 * (y - center_y) / 7.7
    )
    angle = np.arctan2(y - center_y, x - center_x) - np.deg2rad(23.0)
    spiders = (np.abs(np.sin(2.0 * angle)) * radius < 0.18) & (radius > 2.0)
    image[spiders] *= 0.05
    return image + 25.0 + rng.normal(0.0, 3.0, image.shape)


def test_subpixel_center_with_spiders_gradient_and_noise():
    rng = np.random.default_rng(12)
    errors = []
    for index in range(12):
        center_x = 15.5 + rng.uniform(-1.2, 1.2)
        center_y = 15.5 + rng.uniform(-1.2, 1.2)
        result = estimate_pupil_center_subpixel(
            _synthetic_at_pupil(center_x, center_y, index)
        )
        errors.append(np.hypot(
            result["center_x"] - center_x,
            result["center_y"] - center_y,
        ))
        assert result["fit_success"]
        assert result["angular_coverage_fraction"] >= 0.75

    assert np.percentile(errors, 95) < 0.05


def test_both_references_use_clear_pupil_normalization():
    y, x = np.indices((32, 32), dtype=float)
    pupil = (
        (np.hypot(x - 15.3, y - 15.7) < 8.0)
        & (np.hypot(x - 15.3, y - 15.7) > 2.0)
    )
    clear = np.full((32, 32), 10.0)
    clear[pupil] = 15.0
    masked = np.full((32, 32), 10.0)
    masked[pupil] = 20.0

    clear, masked, interior, scale = WRAPPER.normalize_reference_pair(
        clear,
        masked,
        clear_background=10.0,
        masked_background=10.0,
    )

    assert scale == 5.0
    np.testing.assert_allclose(np.mean(clear[interior]), 1.0)
    np.testing.assert_allclose(np.mean(masked[interior]), 2.0)


def test_fractional_frame_center_is_not_rounded():
    y, x = np.indices((64, 64), dtype=float)
    image = 100.0 * y + x
    frame = WRAPPER.extract_frame(image, (32, 32), (15.25, 16.125))
    expected = 100.0 * 31.375 + 31.25
    np.testing.assert_allclose(frame[16, 15], expected)
