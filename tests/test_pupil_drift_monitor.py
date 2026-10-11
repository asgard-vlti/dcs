"""Offline tests for pupil drift correction without instrument hardware."""

import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, call, patch

import numpy as onp
import zmq

from pyeng_heimdallr.tip_tilt_sensor import pupil_drift_monitor as monitor


class PupilDriftMonitorTest(unittest.TestCase):
    def test_visibility_average_uses_each_frame(self):
        hmd = Mock()
        dstream = Mock()
        dstream.get_data.side_effect = [
            onp.array([1001.0]),
            onp.array([1002.0]),
            onp.array([1003.0]),
        ]
        hmd.get_raw_cvis.side_effect = lambda _, img, full: img.astype(complex)

        v2 = monitor.average_visibility(hmd, dstream, nim=3)

        onp.testing.assert_allclose(v2, [14 / 3])
        self.assertEqual(dstream.get_data.call_count, 3)

    def test_move_threshold_is_strict_distance_and_reverses_error(self):
        errors = onp.array(
            [[0.05, 0.0], [0.03, 0.04], [-0.06, 0.02], [0.0, -0.051]]
        )

        commands = monitor.pupil_move_commands(errors)

        self.assertEqual(
            commands,
            [
                (3, "mv_pup c_red_one_focus 3 0.060000 -0.020000"),
                (4, "mv_pup c_red_one_focus 4 -0.000000 0.051000"),
            ],
        )

    def test_invalid_errors_reject_all_moves(self):
        for errors in (
            onp.zeros((3, 2)),
            onp.array([[0.1, 0.0], [onp.nan, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            onp.array([[0.1, 0.0], [onp.inf, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        ):
            with self.subTest(errors=errors):
                with self.assertRaisesRegex(ValueError, "four finite"):
                    monitor.pupil_move_commands(errors)

    def test_dry_run_does_not_open_mds_socket(self):
        commands = [(1, "mv_pup c_red_one_focus 1 -0.100000 0.020000")]
        output = io.StringIO()
        with patch.object(monitor.zmq, "Context") as context, redirect_stdout(output):
            monitor.send_pupil_moves(commands, dry_run=True)

        context.assert_not_called()
        self.assertIn("DRY RUN: mv_pup c_red_one_focus 1", output.getvalue())

    def test_mds_acknowledges_each_move(self):
        commands = [(1, "first"), (2, "second")]
        context = Mock()
        socket = context.socket.return_value
        socket.recv_string.side_effect = ["ACK: moved\n", "ACK: moved\n"]

        with patch.object(monitor.zmq, "Context", return_value=context):
            monitor.send_pupil_moves(commands)

        socket.send_string.assert_has_calls([call("first"), call("second")])
        self.assertEqual(socket.recv_string.call_count, 2)
        socket.close.assert_called_once()
        context.term.assert_called_once()

    def test_nack_stops_remaining_moves_and_reports_partial_success(self):
        commands = [(1, "first"), (2, "second"), (4, "third")]
        context = Mock()
        socket = context.socket.return_value
        socket.recv_string.side_effect = ["ACK: moved\n", "NACK: not moved\n"]

        with patch.object(monitor.zmq, "Context", return_value=context):
            with self.assertRaisesRegex(RuntimeError, r"confirmed moved beams: \[1\]"):
                monitor.send_pupil_moves(commands)

        socket.send_string.assert_has_calls([call("first"), call("second")])
        self.assertEqual(socket.send_string.call_count, 2)
        socket.close.assert_called_once()
        context.term.assert_called_once()

    def test_timeout_stops_remaining_moves_and_reports_partial_success(self):
        commands = [(1, "first"), (2, "second"), (4, "third")]
        context = Mock()
        socket = context.socket.return_value
        socket.recv_string.side_effect = ["ACK: moved\n", zmq.Again()]

        with patch.object(monitor.zmq, "Context", return_value=context):
            with self.assertRaisesRegex(RuntimeError, r"confirmed moved beams: \[1\]"):
                monitor.send_pupil_moves(commands)

        self.assertEqual(socket.send_string.call_count, 2)
        socket.close.assert_called_once()
        context.term.assert_called_once()

    def test_main_wires_default_live_and_dry_run(self):
        hmd = Mock()
        hmd.dense.kpi.UVC = onp.array([[0.0, 0.0]])
        hmd.sparse.kpi.VAC = onp.zeros((4, 2))
        hmd.get_raw_cvis.return_value = onp.array([0.1, 0.1])
        hmd.infer_pupil_model.return_value = onp.array(
            [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )
        sensor_module = types.ModuleType("hmd_tts")
        sensor_module.HMD_TTS = Mock(return_value=hmd)

        from xaosim import shmlib

        dstream = Mock()
        dstream.get_data.return_value = onp.array([1001.0])
        with (
            patch.dict(sys.modules, {"hmd_tts": sensor_module}),
            patch.object(shmlib, "shm", return_value=dstream),
            patch.object(monitor.plt, "subplots", return_value=(Mock(), Mock())),
            patch.object(monitor.plt, "close"),
            patch.object(monitor, "log"),
            patch.object(monitor, "send_pupil_moves") as send,
        ):
            monitor.main([])
            send.assert_called_with(
                [(1, "mv_pup c_red_one_focus 1 -0.100000 -0.000000")],
                dry_run=False,
            )
            monitor.main(["--dry-run"])
            send.assert_called_with(
                [(1, "mv_pup c_red_one_focus 1 -0.100000 -0.000000")],
                dry_run=True,
            )
            self.assertEqual(dstream.get_data.call_count, 100)


if __name__ == "__main__":
    unittest.main()
