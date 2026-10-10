"""Offline tests for the camera recovery sequence; no instrument connections."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, call, patch

from dcs.cmd_scripts import power_cycle_camera as camera


class PowerCycleCameraTest(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.addCleanup(self.output.close)
        self.stdout = redirect_stdout(self.output)
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def test_recovery_sequence_and_final_crop_check(self):
        events = []
        pdu = Mock()
        pdu.connect.return_value = True
        pdu_class = Mock(return_value=pdu)
        kaya_class = Mock()

        def cli(command):
            events.append(f"cli:{command}")
            return "active" if command == "cropping" else ""

        with (
            patch.object(camera, "preflight", return_value=(pdu_class, kaya_class)),
            patch.object(camera, "send_cli", side_effect=cli),
            patch.object(camera, "stop_server", side_effect=lambda: events.append("stop")),
            patch.object(camera, "cycle_pdu", side_effect=lambda _: events.append("pdu")),
            patch.object(camera, "start_server", side_effect=lambda: events.append("start")),
            patch.object(
                camera,
                "monitor_camera",
                side_effect=lambda deadline, cls: events.append("monitor"),
            ) as monitor,
            patch.object(camera.time, "monotonic", return_value=100),
        ):
            camera.power_cycle_camera()

        self.assertEqual(
            events,
            [
                "cli:set cooling off",
                "cli:shutdown",
                "stop",
                "pdu",
                "start",
                "cli:set cooling on",
                "monitor",
                "stop",
                "start",
                "cli:cropping",
            ],
        )
        monitor.assert_called_once_with(100 + 15 * 60, kaya_class)
        pdu.close.assert_called_once()

    def test_pre_power_failure_does_not_cycle_outlet(self):
        pdu = Mock()
        pdu.connect.return_value = True
        with (
            patch.object(camera, "preflight", return_value=(Mock(return_value=pdu), Mock())),
            patch.object(camera, "send_cli", side_effect=RuntimeError("no reply")),
            patch.object(camera, "cycle_pdu") as cycle,
        ):
            with self.assertRaisesRegex(RuntimeError, "no reply"):
                camera.power_cycle_camera()
        cycle.assert_not_called()
        pdu.close.assert_called_once()

    def test_server_exit_failure_does_not_cycle_outlet(self):
        pdu = Mock()
        pdu.connect.return_value = True
        with (
            patch.object(camera, "preflight", return_value=(Mock(return_value=pdu), Mock())),
            patch.object(camera, "send_cli", return_value=""),
            patch.object(camera, "stop_server", side_effect=RuntimeError("server still running")),
            patch.object(camera, "cycle_pdu") as cycle,
        ):
            with self.assertRaisesRegex(RuntimeError, "server still running"):
                camera.power_cycle_camera()
        cycle.assert_not_called()
        pdu.close.assert_called_once()

    def test_outlet_is_restored_after_interruption(self):
        pdu = Mock()
        with (
            patch.object(camera, "wait_for_outlet") as wait,
            patch.object(camera.time, "sleep", side_effect=KeyboardInterrupt),
        ):
            with self.assertRaises(KeyboardInterrupt):
                camera.cycle_pdu(pdu)
        self.assertEqual(
            pdu.switch_outlet_status.call_args_list,
            [call(6, "off"), call(6, "on")],
        )
        self.assertEqual(wait.call_args_list, [call(pdu, "off"), call(pdu, "on")])

    def test_outlet_is_restored_after_off_verification_failure(self):
        pdu = Mock()
        with patch.object(
            camera, "wait_for_outlet", side_effect=[RuntimeError("not off"), None]
        ) as wait:
            with self.assertRaisesRegex(RuntimeError, "not off"):
                camera.cycle_pdu(pdu)
        self.assertEqual(
            pdu.switch_outlet_status.call_args_list,
            [call(6, "off"), call(6, "on")],
        )
        self.assertEqual(wait.call_args_list, [call(pdu, "off"), call(pdu, "on")])

    def test_monitor_restarts_kaya_before_sampling_and_stops_at_operational(self):
        events = []
        statuses = iter(["isbeingcooled", "operational"])
        with (
            patch.object(
                camera,
                "send_command",
                side_effect=lambda command: events.append(command) or 85.0,
            ),
            patch.object(
                camera,
                "send_cli",
                side_effect=lambda command: events.append(command) or next(statuses),
            ),
            patch.object(camera, "restart_kaya", side_effect=lambda cls: events.append("kaya")),
            patch.object(camera.time, "monotonic", return_value=0),
            patch.object(camera.time, "sleep") as sleep,
        ):
            camera.monitor_camera(900, Mock())
        self.assertEqual(
            events,
            ["kaya", "get_det_temp", "status", "get_det_temp", "status"],
        )
        sleep.assert_called_once_with(5)

    def test_monitor_times_out_after_15_minutes(self):
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        with (
            patch.object(camera, "send_command", return_value=91.5),
            patch.object(camera, "send_cli", return_value="isbeingcooled"),
            patch.object(camera, "restart_kaya") as restart,
            patch.object(camera.time, "monotonic", side_effect=lambda: now[0]),
            patch.object(camera.time, "sleep", side_effect=sleep),
        ):
            with self.assertRaisesRegex(RuntimeError, "within 15 minutes") as error:
                camera.monitor_camera(900, Mock())
        self.assertEqual(now[0], 900)
        self.assertIn("91.5", str(error.exception))
        self.assertIn("isbeingcooled", str(error.exception))
        restart.assert_called_once()

    def test_operational_status_after_deadline_is_too_late(self):
        now = [899.0]

        def late_status(_):
            now[0] = 901.0
            return "operational"

        with (
            patch.object(camera, "send_command", return_value=85.0),
            patch.object(camera, "send_cli", side_effect=late_status),
            patch.object(camera, "restart_kaya"),
            patch.object(camera.time, "monotonic", side_effect=lambda: now[0]),
        ):
            with self.assertRaisesRegex(RuntimeError, "within 15 minutes"):
                camera.monitor_camera(900, Mock())

    def test_server_exit_waits_for_lock_release(self):
        with (
            patch.object(camera, "send_command", return_value="Exiting!") as send,
            patch.object(camera, "wait_for_server_stop") as wait,
        ):
            camera.stop_server()
        send.assert_called_once_with("exit")
        wait.assert_called_once()

    def test_server_start_waits_for_valid_status(self):
        responses = iter(["", {"cam_status": "running"}])
        with (
            patch.object(camera.subprocess, "run") as run,
            patch.object(camera, "send_command", side_effect=lambda _: next(responses)) as send,
            patch.object(camera.time, "monotonic", return_value=0),
            patch.object(camera.time, "sleep") as sleep,
        ):
            camera.start_server()
        run.assert_called_once()
        self.assertEqual(send.call_args_list, [call("status"), call("status")])
        sleep.assert_called_once_with(1)

    def test_kaya_restart_accepts_powered_on_status(self):
        kaya = Mock()
        kaya.get_status.return_value = "0"
        kaya.client = None
        with patch.object(camera.subprocess, "run") as run:
            camera.restart_kaya(Mock(return_value=kaya))
        run.assert_called_once_with(["restart-kaya"], check=True, timeout=45)

    def test_kaya_restart_rejects_powered_off_status(self):
        kaya = Mock()
        kaya.get_status.return_value = "1"
        kaya.client = None
        with patch.object(camera.subprocess, "run"):
            with self.assertRaisesRegex(RuntimeError, "did not report powered on"):
                camera.restart_kaya(Mock(return_value=kaya))

    def test_final_crop_failure_is_reported(self):
        pdu = Mock()
        pdu.connect.return_value = True
        with (
            patch.object(camera, "preflight", return_value=(Mock(return_value=pdu), Mock())),
            patch.object(
                camera,
                "send_cli",
                side_effect=lambda command: "inactive" if command == "cropping" else "",
            ),
            patch.object(camera, "stop_server"),
            patch.object(camera, "cycle_pdu"),
            patch.object(camera, "start_server"),
            patch.object(camera, "monitor_camera"),
        ):
            with self.assertRaisesRegex(RuntimeError, "cropping is not active"):
                camera.power_cycle_camera()


if __name__ == "__main__":
    unittest.main()
