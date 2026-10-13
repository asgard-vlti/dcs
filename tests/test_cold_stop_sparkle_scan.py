"""Offline checks for the cold-stop sparkle flux helper."""

import threading
import time
import unittest
from unittest.mock import patch

import numpy as onp
from PyQt5 import QtWidgets

from minimal_baldr_python_rtc.experiments import cold_stop_sparkle_scan as scan


class FakeController:
    def __init__(self, beam, events):
        self.events = events
        events.append(("open DM", beam))

    def set_enabled(self, enabled, distance, angle, amplitude):
        self.events.append(("set DM", enabled, distance, angle, amplitude))

    def close(self):
        self.events.append(("close DM",))


class FakeCamera:
    def __init__(self, beam, events, fail_at=None):
        self.events = events
        self.calls = 0
        self.fail_at = fail_at
        events.append(("open camera", beam))

    def read_fresh(self, cancelled):
        self.calls += 1
        if self.calls == self.fail_at:
            raise TimeoutError("camera stalled")
        self.events.append(("frame", self.calls))
        return onp.full((2, 3), self.calls, dtype=onp.float64)

    def close(self):
        self.events.append(("close camera",))


class ScanAcquisitionTest(unittest.TestCase):
    def test_off_then_full_sweep_with_fresh_frame_averages_and_cleanup(self):
        events = []
        progress = []
        distances, fluxes = scan.acquire_scan(
            2,
            30.0,
            0.02,
            controller_factory=lambda beam: FakeController(beam, events),
            camera_factory=lambda beam: FakeCamera(beam, events),
            sleep_fn=lambda delay: events.append(("settle", delay)),
            on_progress=lambda count, total: progress.append((count, total)),
        )

        onp.testing.assert_array_equal(distances, scan.SEPARATIONS)
        self.assertEqual(len(distances), 51)
        self.assertEqual(distances[-1], 5.0)
        self.assertEqual(len(fluxes), 51)
        self.assertEqual(fluxes[0], 6 * onp.mean(onp.arange(3, 19)))
        self.assertEqual(fluxes[1], 6 * onp.mean(onp.arange(21, 37)))
        self.assertEqual(progress[-1], (51, 51))
        self.assertEqual(
            [event[:3] for event in events if event[0] == "set DM"][:2],
            [("set DM", False, 0.1), ("set DM", True, 0.1)],
        )
        self.assertEqual(
            [event[:3] for event in events if event[0] == "set DM"][-1],
            ("set DM", True, 5.0),
        )
        self.assertEqual(events[-2:], [("close camera",), ("close DM",)])
        self.assertEqual(sum(event[0] == "frame" for event in events), 51 * 18)

    def test_error_and_cancellation_close_both_streams(self):
        for cancelled, fail_at in ((threading.Event(), 4), (threading.Event(), None)):
            with self.subTest(fail_at=fail_at):
                events = []
                if fail_at is None:
                    cancelled.set()
                with self.assertRaises(scan.ScanCancelled if fail_at is None else TimeoutError):
                    scan.acquire_scan(
                        1,
                        0.0,
                        0.02,
                        cancelled=cancelled,
                        controller_factory=lambda beam: FakeController(beam, events),
                        camera_factory=lambda beam: FakeCamera(beam, events, fail_at),
                        sleep_fn=lambda delay: None,
                        separations=(0.0, 0.1),
                    )
                self.assertEqual(events[-2:], [("close camera",), ("close DM",)])


class FakeStream:
    def __init__(self, counters):
        self.counters = iter(counters)
        self.counter = next(self.counters)
        self.closed = False

    def get_counter(self):
        return self.counter

    def get_data(self, check, timeout):
        self.counter = next(self.counters)
        return onp.full((2, 2), self.counter)

    def close(self, erase_file=False):
        self.closed = True


class CameraReaderTest(unittest.TestCase):
    def test_missing_camera_stream_has_clear_error(self):
        class EmptyStream:
            empty = True

        with self.assertRaisesRegex(FileNotFoundError, "baldr1.im.shm"):
            scan.CameraReader(1, lambda path, nosem: EmptyStream())

    def test_skips_stale_frames_and_reads_next_counter(self):
        stream = FakeStream((10, 10, 11))
        reader = scan.CameraReader(3, lambda path, nosem: stream)
        try:
            frame = reader.read_fresh(threading.Event())
            onp.testing.assert_array_equal(frame, onp.full((2, 2), 11))
            self.assertEqual(reader.counter, 11)
        finally:
            reader.close()
        self.assertTrue(stream.closed)

    def test_counter_reset_and_stall_are_rejected(self):
        stream = FakeStream((10, 9))
        reader = scan.CameraReader(1, lambda path, nosem: stream)
        try:
            with self.assertRaisesRegex(RuntimeError, "counter reset"):
                reader.read_fresh(threading.Event())
        finally:
            reader.close()

        stream = FakeStream((10, 10))
        reader = scan.CameraReader(1, lambda path, nosem: stream)
        try:
            with patch.object(scan.time, "monotonic", side_effect=(0.0, 0.0, 3.0)):
                with self.assertRaisesRegex(TimeoutError, "no fresh frame"):
                    reader.read_fresh(threading.Event())
        finally:
            reader.close()


class ScanWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_repeated_runs_have_different_colors_and_clear_lines(self):
        window = scan.ScanWindow(2)
        try:
            def fake_scan(*args, **kwargs):
                return onp.array((0.0, 5.0)), onp.array((100.0, 80.0))

            with patch.object(scan, "acquire_scan", side_effect=fake_scan):
                for expected_count in (1, 2):
                    window._start_scan()
                    self.assertFalse(window.run_button.isEnabled())
                    deadline = time.monotonic() + 2.0
                    while window._thread is not None and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(0.001)
                    self.app.processEvents()
                    self.assertIsNone(window._thread)
                    self.assertEqual(len(window._curves), expected_count)
                    self.assertTrue(window.run_button.isEnabled())

            first = window._curves[0].opts["pen"].color().name()
            second = window._curves[1].opts["pen"].color().name()
            self.assertNotEqual(first, second)
            self.assertEqual(len(window.legend.items), 2)
            window._clear_lines()
            self.assertFalse(window._curves)
            self.assertFalse(window.legend.items)
        finally:
            window.close()

    def test_close_cancels_active_scan(self):
        window = scan.ScanWindow(1)
        started = threading.Event()
        stopped = threading.Event()

        def fake_scan(*args, cancelled, **kwargs):
            started.set()
            try:
                while not cancelled.wait(0.01):
                    pass
                raise scan.ScanCancelled()
            finally:
                stopped.set()

        try:
            with patch.object(scan, "acquire_scan", side_effect=fake_scan):
                window._start_scan()
                deadline = time.monotonic() + 2.0
                while not started.is_set() and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.001)
                self.assertTrue(started.is_set())
                self.assertTrue(window.close())
                self.assertTrue(stopped.is_set())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
