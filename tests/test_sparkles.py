"""Offline checks for the static DM sparkles utility."""

import unittest
from unittest.mock import patch

import numpy as onp
from PyQt5 import QtWidgets

from dcs.cmd_scripts import sparkles


class SparkleMapTest(unittest.TestCase):
    def test_map_is_piston_free_pinned_and_peak_limited(self):
        for distance, angle, amplitude in (
            (0.1, 0.0, 0.2),
            (2.375, 27.5, 0.08),
            (5.0, 0.0, 0.2),
            (5.0, 90.0, 0.2),
        ):
            with self.subTest(distance=distance, angle=angle):
                pattern = sparkles.make_sparkle_map(distance, angle, amplitude)
                self.assertEqual(pattern.shape, (12, 12))
                self.assertTrue(onp.all(onp.isfinite(pattern)))
                onp.testing.assert_array_equal(pattern[~sparkles.ACTIVE], 0)
                self.assertAlmostEqual(float(pattern[sparkles.ACTIVE].mean()), 0, places=14)
                self.assertAlmostEqual(
                    float(onp.max(onp.abs(pattern[sparkles.ACTIVE]))), amplitude, places=14
                )
                onp.testing.assert_allclose(pattern[0, 1:-1], pattern[1, 1:-1])
                onp.testing.assert_allclose(pattern[-1, 1:-1], pattern[-2, 1:-1])
                onp.testing.assert_allclose(pattern[1:-1, 0], pattern[1:-1, 1])
                onp.testing.assert_allclose(pattern[1:-1, -1], pattern[1:-1, -2])

    def test_fractional_separation_and_orthogonal_angle_are_preserved(self):
        distance = 2.375
        theta = onp.deg2rad(30)
        pattern = sparkles.make_sparkle_map(distance, 30, 0.1)[1:-1, 1:-1].ravel()
        grid = sparkles.make_hc_act_grid()
        x = onp.asarray(grid.x).reshape(12, 12)[1:-1, 1:-1].ravel()
        y = onp.asarray(grid.y).reshape(12, 12)[1:-1, 1:-1].ravel()

        def fit_residual(radial_distance):
            frequencies = (
                (radial_distance * onp.cos(theta), radial_distance * onp.sin(theta)),
                (-radial_distance * onp.sin(theta), radial_distance * onp.cos(theta)),
            )
            basis = [onp.ones_like(x)]
            for fx, fy in frequencies:
                phase = 2 * onp.pi * (fx * x + fy * y)
                basis.extend((onp.cos(phase), onp.sin(phase)))
            matrix = onp.column_stack(basis)
            coefficients, *_ = onp.linalg.lstsq(matrix, pattern, rcond=None)
            return onp.linalg.norm(pattern - matrix @ coefficients), coefficients

        residual, coefficients = fit_residual(distance)
        self.assertLess(residual, 1e-12)
        self.assertGreater(onp.linalg.norm(coefficients[1:3]), 0.01)
        self.assertGreater(onp.linalg.norm(coefficients[3:5]), 0.01)
        self.assertGreater(fit_residual(2.0)[0], 0.1)
        self.assertGreater(fit_residual(3.0)[0], 0.1)

    def test_linearized_pupil_has_four_spots_at_requested_separation(self):
        grid = sparkles.make_hc_act_grid()
        x = onp.asarray(grid.x).reshape(12, 12)
        y = onp.asarray(grid.y).reshape(12, 12)
        pupil = x * x + y * y <= 0.5**2
        pattern = sparkles.make_sparkle_map(3.0, 30.0, 0.1)
        focal = onp.fft.fftshift(onp.abs(onp.fft.fft2(pupil * pattern, s=(512, 512))) ** 2)
        frequency = onp.fft.fftshift(onp.fft.fftfreq(512, d=0.1))

        theta = onp.deg2rad(30.0)
        first = (3 * onp.cos(theta), 3 * onp.sin(theta))
        second = (-3 * onp.sin(theta), 3 * onp.cos(theta))
        for fx, fy in (first, (-first[0], -first[1]), second, (-second[0], -second[1])):
            with self.subTest(fx=fx, fy=fy):
                neighborhood = (
                    (frequency[:, None] - fy) ** 2
                    + (frequency[None, :] - fx) ** 2
                    < 0.7**2
                )
                row, column = onp.unravel_index(
                    onp.argmax(onp.where(neighborhood, focal, 0)), focal.shape
                )
                self.assertLess(onp.hypot(frequency[column] - fx, frequency[row] - fy), 0.2)

    def test_zero_amplitude_and_invalid_inputs(self):
        onp.testing.assert_array_equal(sparkles.make_sparkle_map(2.5, 45, 0), 0)
        for values in (
            (0, 0, 0.1),
            (5.1, 0, 0.1),
            (2, -1, 0.1),
            (2, 91, 0.1),
            (2, 0, -0.01),
            (2, 0, 0.21),
            (onp.nan, 0, 0.1),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                sparkles.make_sparkle_map(*values)


class FakeStream:
    def __init__(self, path):
        self.path = path
        self.empty = False
        self.data = None
        self.writes = []
        self.posts = []
        self.closed = False

    def set_data(self, data):
        self.data = onp.asarray(data).copy()
        self.writes.append(self.data)

    def post_sems(self, index):
        self.posts.append(index)

    def close(self, erase_file=False):
        self.closed = True


class DmChannelTest(unittest.TestCase):
    def test_writes_only_disp03_and_posts_combined_semaphore_one(self):
        streams = {}

        def make_stream(path, nosem):
            self.assertFalse(nosem)
            streams[path] = FakeStream(path)
            return streams[path]

        with patch.object(sparkles.Path, "is_file", return_value=True), patch(
            "xaosim.shmlib.shm", side_effect=make_stream
        ):
            channel = sparkles.DmChannel(2)
        expected = sparkles.make_sparkle_map(2.25, 30, 0.1)
        channel.write(expected)
        channel.clear()
        channel.close()

        self.assertEqual(
            set(streams),
            {"/dev/shm/dm2disp03.im.shm", "/dev/shm/dm2.im.shm"},
        )
        onp.testing.assert_allclose(streams["/dev/shm/dm2disp03.im.shm"].writes[0], expected)
        onp.testing.assert_array_equal(streams["/dev/shm/dm2disp03.im.shm"].data, 0)
        self.assertEqual(streams["/dev/shm/dm2.im.shm"].posts, [1, 1])
        self.assertTrue(all(stream.closed for stream in streams.values()))

    def test_missing_stream_fails_before_opening_shared_memory(self):
        with patch.object(sparkles.Path, "is_file", return_value=False), patch(
            "xaosim.shmlib.shm"
        ) as open_shm:
            with self.assertRaises(FileNotFoundError):
                sparkles.DmChannel(1)
        open_shm.assert_not_called()


class FakeChannel:
    def __init__(self, beam, events):
        self.beam = beam
        self.events = events

    def write(self, pattern):
        self.events.append((self.beam, "write", onp.asarray(pattern).copy()))

    def clear(self):
        self.events.append((self.beam, "clear"))

    def close(self):
        self.events.append((self.beam, "close"))


class ControllerTest(unittest.TestCase):
    def test_lifecycle_and_beam_change(self):
        events = []
        controller = sparkles.SparklesController(
            1, lambda beam: FakeChannel(beam, events)
        )
        self.assertEqual(events, [(1, "clear")])

        controller.update(2, 0, 0.1)
        self.assertEqual(events, [(1, "clear")])
        controller.set_enabled(True, 2.375, 30, 0.1)
        controller.update(2.5, 30, 0.1)
        controller.set_beam(2, 2.5, 30, 0.1)
        controller.set_enabled(False, 2.5, 30, 0.1)
        controller.close()

        self.assertEqual(
            [(event[0], event[1]) for event in events],
            [
                (1, "clear"), (1, "write"), (1, "write"), (1, "clear"),
                (1, "close"), (2, "write"), (2, "clear"), (2, "clear"),
                (2, "close"),
            ],
        )
        self.assertFalse(controller.enabled)


class SparklesWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_offscreen_controls_accept_fractional_values(self):
        events = []
        window = sparkles.SparklesWindow(2, lambda beam: FakeChannel(beam, events))
        try:
            self.assertFalse(window.on_input.isChecked())
            self.assertEqual(window.distance_input.decimals(), 3)
            self.assertEqual(window.amplitude_input.decimals(), 3)
            window.distance_input.setValue(2.375)
            window.angle_input.setValue(37.5)
            window.amplitude_input.setValue(0.125)
            self.assertEqual(len(events), 1)
            window.on_input.setChecked(True)
            self.assertEqual(events[-1][:2], (2, "write"))
            window.beam_input.setCurrentIndex(2)
            self.assertEqual(events[-3][:2], (2, "clear"))
            self.assertEqual(events[-1][:2], (3, "write"))
            window.on_input.setChecked(False)
            self.assertEqual(events[-1], (3, "clear"))
        finally:
            window.close()
        self.assertEqual(events[-2:], [(3, "clear"), (3, "close")])

    def test_dm_error_is_shown_and_on_state_reverts(self):
        class FailingChannel(FakeChannel):
            def write(self, pattern):
                raise OSError("test DM write failure")

        events = []
        window = sparkles.SparklesWindow(1, lambda beam: FailingChannel(beam, events))
        try:
            window.on_input.setChecked(True)
            self.assertFalse(window.on_input.isChecked())
            self.assertIn("test DM write failure", window.status.text())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
