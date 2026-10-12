"""Offline checks for the standard-mode Baldr fitting acquisition."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as onp

from minimal_baldr_python_rtc.experiments import measure_baldr_standard_for_model as measure


class FakeSHM:
    def __init__(self, data):
        self.data = onp.asarray(data, dtype=onp.float64).copy()
        self.writes = []
        self.posts = 0

    def get_data(self):
        return self.data.copy()

    def set_data(self, data):
        self.data = onp.asarray(data, dtype=onp.float64).copy()
        self.writes.append(self.data.copy())

    def post_sems(self, count):
        self.posts += count


class FakeDM:
    def __init__(self):
        self.main_chn = 2
        self.basis = onp.zeros((144, 2))
        self.basis[3 * 12 + 3, 0] = 1
        self.basis[8 * 12 + 8, 1] = 1
        self.shms = [FakeSHM(onp.zeros(144)) for _ in range(4)]
        self.shms[0].data[:] = 0.2
        self.shms[2].data[:] = 0.01
        self.shm0 = FakeSHM(onp.full(144, 0.21))


class FakeCam:
    imsize = (2, 2)

    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def take_stack(self, count, subtract_dark=True):
        self.calls.append((count, subtract_dark))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("camera frame failure")
        return onp.full((count, *self.imsize), len(self.calls), dtype=onp.uint16)


class FakeMDS:
    def __init__(self):
        self.positions = {"BMX": 2200.0, "BMY": 3000.0, "BTP": 0.1, "BTT": -0.1}
        self.commands = []
        self.shutter_position = None

    def send(self, command):
        self.commands.append(command)
        parts = command.split()
        if parts[:2] == ["b_shut", "close"]:
            self.shutter_position = self.positions["BMY"]
            self.positions["BMY"] = measure.SHUTTER_BMY_UM
            return ""
        if parts[:2] == ["b_shut", "open"]:
            if self.shutter_position is not None:
                self.positions["BMY"] = self.shutter_position
                self.shutter_position = None
            return ""
        if parts[0] == "state":
            return "No error SHUTTERED" if self.shutter_position is not None else "No error"
        if parts[0] == "read":
            return str(self.positions[parts[1][:-1]])
        if parts[0] == "moveabs":
            self.positions[parts[1][:-1]] = float(parts[2])
            return "ACK"
        raise AssertionError(f"Unexpected MDS command: {command}")


class MeasureBaldrTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = Path(self.temp_dir.name)
        self.dm = FakeDM()
        self.mds = FakeMDS()
        self.config = measure.AcquisitionConfig(
            dark_frames=3,
            pupil_frames=2,
            reference_frames=4,
            probe_frames=2,
            probe_repeats=1,
            discard_frames=1,
            dm_settle_s=0,
            motor_settle_s=0,
            motor_poll_s=0,
            motor_timeout_s=1,
            mode_count=2,
        )
        self.sleep_patch = mock.patch.object(measure.time, "sleep", return_value=None)
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def assert_restored(self):
        self.assertEqual(self.mds.positions["BMY"], 3000)
        self.assertEqual(self.mds.positions["BTP"], 0.1)
        self.assertEqual(self.mds.positions["BTT"], -0.1)
        self.assertIsNone(self.mds.shutter_position)
        onp.testing.assert_array_equal(self.dm.shms[2].data, onp.full(144, 0.01))
        onp.testing.assert_array_equal(self.dm.shms[0].data, onp.full(144, 0.2))
        onp.testing.assert_array_equal(self.dm.shms[1].data, onp.zeros(144))
        onp.testing.assert_array_equal(self.dm.shms[3].data, onp.zeros(144))

    def test_motor_move_waits_two_seconds_after_reaching_position(self):
        config = measure.AcquisitionConfig()
        self.assertGreaterEqual(config.motor_settle_s, 2.0)
        measure.move_motor(self.mds, 2, "BTP", 0.13, config)
        measure.move_motor(self.mds, 2, "BTT", -0.07, config)
        self.assertEqual(
            [call.args[0] for call in measure.time.sleep.call_args_list],
            [config.motor_poll_s, config.motor_settle_s] * 2,
        )

    def test_complete_acquisition_preserves_processed_frames_and_probe_order(self):
        cam = FakeCam()
        output = measure.run_and_save(
            2,
            self.dm,
            cam,
            self.mds,
            output_dir=self.output_dir,
            config=self.config,
            rtc_status={"servo": "off", "wants_to_close": False},
            camera_status={"fps": 1000, "gain": 5},
        )
        self.assert_restored()
        self.assertTrue(all(not subtract_dark for _, subtract_dark in cam.calls))
        self.assertLess(
            self.mds.commands.index("b_shut close 2"),
            self.mds.commands.index("b_shut open 2"),
        )
        self.assertIn("moveabs BMY2 2800.000000", self.mds.commands)
        self.assertTrue(self.dm.shms[2].writes)
        for command in self.dm.shms[2].writes:
            self.assertAlmostEqual(float(command.mean()), 0.01)

        with onp.load(output, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"]))
            self.assertTrue(bool(saved["complete"]))
            self.assertTrue(metadata["camera_server_dark_subtracted"])
            self.assertEqual(metadata["camera_status"]["gain"], 5)
            self.assertEqual(saved["dark_frames"].shape, (3, 2, 2))
            self.assertEqual(saved["pupil_dither_frames"].shape, (9, 2, 2, 2))
            self.assertEqual(saved["masked_reference_frames"].shape, (4, 2, 2))
            self.assertEqual(saved["pupil_fourier_frames"].shape, (4, 2, 2, 2))
            self.assertEqual(saved["masked_fourier_frames"].shape, (4, 2, 2, 2))
            self.assertEqual(saved["masked_poke_frames"].shape, (8, 2, 2, 2))
            onp.testing.assert_array_equal(saved["pupil_fourier_index"], [0, 0, 1, 1])
            onp.testing.assert_array_equal(saved["pupil_fourier_sign"], [-1, 1, -1, 1])
            onp.testing.assert_array_equal(saved["masked_poke_sign"], [-1, 1] * 4)
            onp.testing.assert_allclose(
                saved["pupil_dither_offsets_deg"],
                [(p, t) for p in (-0.03, 0, 0.03) for t in (-0.03, 0, 0.03)],
            )
            self.assertEqual(saved["pupil_dither_positions"].shape, (9, 4))
            self.assertAlmostEqual(float(saved["fourier_delta_maps"][0].sum()), 0, places=6)
            self.assertAlmostEqual(float(saved["poke_delta_maps"][0].max()), 0.03 * 143 / 144)

    def test_camera_failure_saves_aligned_partial_data_and_restores_state(self):
        cam = FakeCam(fail_on_call=5)
        with self.assertRaisesRegex(RuntimeError, "camera frame failure"):
            measure.run_and_save(
                3, self.dm, cam, self.mds, output_dir=self.output_dir, config=self.config
            )
        self.assert_restored()
        outputs = list(self.output_dir.glob("*.npz"))
        self.assertEqual(len(outputs), 1)
        with onp.load(outputs[0], allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"]))
            self.assertFalse(bool(saved["complete"]))
            self.assertIn("camera frame failure", metadata["failure"])
            self.assertEqual(saved["pupil_dither_frames"].shape[0], 3)
            self.assertEqual(saved["pupil_dither_offsets_deg"].shape[0], 3)
            self.assertEqual(saved["pupil_dither_positions"].shape[0], 3)

    def test_dark_capture_failure_reopens_shutter(self):
        with self.assertRaisesRegex(RuntimeError, "camera frame failure"):
            measure.run_and_save(
                4,
                self.dm,
                FakeCam(fail_on_call=1),
                self.mds,
                output_dir=self.output_dir,
                config=self.config,
            )
        self.assert_restored()
        self.assertIn("b_shut close 4", self.mds.commands)
        self.assertIn("b_shut open 4", self.mds.commands)
        with onp.load(next(self.output_dir.glob("*.npz")), allow_pickle=False) as saved:
            self.assertFalse(bool(saved["complete"]))
            self.assertEqual(saved["dark_frames"].shape[0], 0)

    def test_pending_servo_aborts_before_motor_motion(self):
        with self.assertRaisesRegex(RuntimeError, "servo is active"):
            measure.run_and_save(
                1,
                self.dm,
                FakeCam(),
                self.mds,
                output_dir=self.output_dir,
                config=self.config,
                rtc_status={"servo": "off", "wants_to_close": True},
            )
        self.assert_restored()
        self.assertEqual(self.mds.commands, [])


if __name__ == "__main__":
    unittest.main()
