from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from baldr_reference.model import generate_references, load_config


@pytest.mark.slow
def test_matches_baldrapp_reference():
    config = load_config("configs/reference_internal.example.json")
    material = Path("src/baldr_reference/Exposed_Ma-N_1405_optical_constants.txt")
    clear, masked, _ = generate_references(config, material)
    with fits.open("tests/data/baldrapp_reference_intensities.fits") as expected:
        assert np.allclose(clear, expected[0].data, rtol=1e-12, atol=1e-10)
        assert np.allclose(masked, expected[1].data, rtol=1e-12, atol=1e-10)
