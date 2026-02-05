# baldr_rtc/io/backend_tests/test_shm_backend.py
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest


def _beam() -> int:
    # Override at runtime with: BALDR_BEAM=3 pytest -q
    return int(os.environ.get("BALDR_BEAM", "1"))


def _have_dev_shm() -> bool:
    return Path("/dev/shm").exists()


def _shm_files_exist(pattern: str) -> bool:
    import glob

    return len(glob.glob(pattern)) > 0


@pytest.fixture(scope="module")
def shm_backend():
    # Import from your package location
    try:
        from baldr_python_rtc.baldr_rtc.io import shm_backend as sb
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Cannot import baldr_rtc.io.shm_backend: {e}")
    if getattr(sb, "shm", None) is None:
        pytest.skip("xaosim.shmlib.shm not importable (xaosim not installed?)")
    if not _have_dev_shm():
        pytest.skip("/dev/shm not present on this system")
    return sb


def test_cmd_2_map2D_inserts_corners(shm_backend):
    cmd = np.arange(140, dtype=float)
    m = shm_backend.ShmDMIO.cmd_2_map2D(cmd, fill=-999.0)
    assert m.shape == (12, 12)

    flat = m.reshape(-1)
    # Insert indices in the 144-vector
    assert flat[0] == -999.0
    assert flat[10] == -999.0
    assert flat[130] == -999.0
    assert flat[140] == -999.0

    # Remove corners and compare to original
    idx = np.ones(144, dtype=bool)
    idx[[0, 10, 130, 140]] = False
    back = flat[idx]
    assert back.shape == (140,)
    assert np.allclose(back, cmd)


def test_shm_camera_get_frame(sm_backend := None):
    # Keep as a standalone test if you want to run it explicitly;
    # otherwise the fixture-based version below is usually enough.
    pass


def test_camera_io_get_frame(shm_backend):
    beam = _beam()
    cam_path = f"/dev/shm/baldr{beam}.im.shm"
    if not Path(cam_path).exists():
        pytest.skip(f"Camera SHM file not found: {cam_path}")

    cam = shm_backend.ShmCameraIO(cam_path, nosem=True, semid=None)

    if getattr(cam, "empty", False):
        pytest.skip(f"Camera SHM exists but reports empty=True: {cam_path}")

    fr = cam.get_frame()
    assert hasattr(fr, "data")
    assert hasattr(fr, "t_s")
    assert hasattr(fr, "frame_id")
    assert isinstance(fr.data, np.ndarray)
    assert fr.data.ndim in (1, 2, 3)  # depends on your producer
    assert isinstance(fr.frame_id, int)
    assert fr.frame_id >= 0

    # sanity: second call should not crash; frame_id should be int
    fr2 = cam.get_frame()
    assert isinstance(fr2.frame_id, int)

    cam.close()


def test_dm_io_init_and_basic_write_readback(shm_backend):
    import glob

    beam = _beam()
    patt = f"/dev/shm/dm{beam}disp*.im.shm"
    combined = f"/dev/shm/dm{beam}.im.shm"

    if not Path(combined).exists():
        pytest.skip(f"Combined DM SHM not found: {combined}")
    if not _shm_files_exist(patt):
        pytest.skip(f"No DM subchannel SHMs found: {patt}")

    dm = shm_backend.ShmDMIO(beam, main_chn=2, nosem=False)

    # Write a deterministic command and read it back from the same subchannel SHM object
    cmd = np.zeros((12, 12), dtype=float)
    cmd[3, 4] = 0.123
    dm.write(cmd)

    # xaosim.shmlib.shm typically supports get_data() or get_latest_data()
    # Your class uses set_data(); for readback we try get_data first.
    sh = dm.shms[dm.main_chn]
    readback = None
    if hasattr(sh, "get_data"):
        readback = np.asarray(sh.get_data())
    elif hasattr(sh, "get_latest_data"):
        readback = np.asarray(sh.get_latest_data())
    elif hasattr(sh, "get_latest_data_slice"):
        readback = np.asarray(sh.get_latest_data_slice())
    else:
        pytest.skip("SHM object has no readable data method (get_data/get_latest_data/...)")

    assert readback.shape == (12, 12)
    assert np.isclose(readback[3, 4], 0.123)

    dm.close()


def test_dm_write_accepts_140_144_12x12(shm_backend):
    beam = _beam()
    patt = f"/dev/shm/dm{beam}disp*.im.shm"
    combined = f"/dev/shm/dm{beam}.im.shm"

    if not Path(combined).exists() or not _shm_files_exist(patt):
        pytest.skip("DM SHM not available in this environment")

    dm = shm_backend.ShmDMIO(beam, main_chn=2, nosem=False)

    # 12x12
    dm.write(np.zeros((12, 12), dtype=float))

    # 140 -> should be converted
    v140 = np.linspace(0.0, 1.0, 140)
    dm.write(v140)

    # 144 -> reshape
    v144 = np.linspace(0.0, 1.0, 144)
    dm.write(v144)

    dm.close()


def test_dm_activate_flat_if_files_present(shm_backend):
    beam = _beam()
    patt = f"/dev/shm/dm{beam}disp*.im.shm"
    combined = f"/dev/shm/dm{beam}.im.shm"
    if not Path(combined).exists() or not _shm_files_exist(patt):
        pytest.skip("DM SHM not available in this environment")

    dm = shm_backend.ShmDMIO(beam, main_chn=2, nosem=False)

    # Only run if the DMShapes dir + expected flat exists
    flat_path = dm.select_flat_cmd()
    if not Path(flat_path).exists():
        pytest.skip(f"Flat file not found (skipping): {flat_path}")

    dm.activate_flat()  # should not raise

    dm.close()