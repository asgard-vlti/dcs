import copy

import numpy as np

from baldr_reference.model import (
    PHASE_MASK_PRESETS,
    load_config,
    make_pupil,
    make_spectrum,
    resolve_source_profile,
)


def config():
    return load_config("configs/reference_internal.example.json")


def test_bandwidth_is_derived_from_limits():
    wavelengths, weights_nm = make_spectrum(config())
    assert wavelengths[[0, -1]].tolist() == [1.5e-6, 1.8e-6]
    assert np.isclose(weights_nm.sum(), 300.0)


def test_temperature_overrides_spectral_type():
    trial = config()
    profile = trial["source"]["profiles"]["internal"]
    profile["spectral_type"] = "G2V"
    profile["temperature_K"] = 3123.0
    assert resolve_source_profile(trial)[2] == 3123.0


def test_all_named_masks_have_physical_dimensions():
    assert set(PHASE_MASK_PRESETS) == {
        "H1", "H2", "H3", "H4", "H5", "J1", "J2", "J3", "J4", "J5"
    }
    assert all(item["depth_um"] > 0 and item["diameter_um"] > 0 for item in PHASE_MASK_PRESETS.values())


def test_named_pupil_ignores_custom_overrides():
    nominal = config()
    nominal["pupil"]["geometry"] = "UT"
    overridden = copy.deepcopy(nominal)
    overridden["pupil"].update(
        central_obstruction=0.7,
        spiders={"count": 1, "thickness": 0.5},
    )
    assert np.array_equal(make_pupil(nominal), make_pupil(overridden))


def test_custom_pupil_uses_symmetric_spiders_and_rotation():
    nominal = config()
    nominal["pupil"].pop("geometry")
    nominal["pupil"].update(
        central_obstruction=0.14,
        spiders={"count": 5, "thickness": 0.02},
    )
    rotated = copy.deepcopy(nominal)
    rotated["pupil"]["rotation_deg"] = 17.0
    assert make_pupil(nominal).shape == (256, 256)
    assert not np.array_equal(make_pupil(nominal), make_pupil(rotated))
