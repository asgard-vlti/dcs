import json

from astropy.io import fits

from baldr_reference.fitting import _registered_extract, measure_pupil, read_reference_fits
from baldr_reference.generate import add_complete_config


def test_complete_config_header_roundtrip():
    config = {"source": {"profile": "internal"}, "values": list(range(50))}
    header = fits.Header()
    add_complete_config(header, config)
    restored = json.loads("".join(header[f"CFG{i:04d}"] for i in range(header["CFGNCHNK"])))
    assert restored == config


def test_reader_accepts_two_frame_cube(tmp_path):
    import numpy as np

    path = tmp_path / "cube.fits"
    fits.PrimaryHDU(np.arange(32).reshape(2, 4, 4)).writeto(path)
    clear, masked = read_reference_fits(path)
    assert clear.shape == masked.shape == (4, 4)
    assert clear[0, 0] == 0
    assert masked[0, 0] == 16


def test_registered_extract_places_model_pupil_at_measured_center():
    import numpy as np

    model = np.zeros((64, 64))
    yy, xx = np.indices(model.shape)
    model[(xx - 31.5) ** 2 + (yy - 31.5) ** 2 <= 7.5**2] = 1.0
    extracted = _registered_extract(model, (32, 32), (31.5, 31.5), (12.5, 18.5))
    measurement = measure_pupil(extracted)
    assert measurement["center_y"] == 12.5
    assert measurement["center_x"] == 18.5
