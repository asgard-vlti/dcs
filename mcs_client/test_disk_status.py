import datetime
import json
import os
import pathlib
import tempfile
import unittest

from mcs_client.disk_status import DiskSavingMonitor
from mcs_client.mcs_client import MCSServer


class DiskSavingMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.now = 1_700_000_000.0
        self.ps_output = ""
        self.monitor = DiskSavingMonitor(
            self.root, process_list=lambda: self.ps_output, clock=lambda: self.now
        )

    def _writer(self, name, pid=100, elapsed=60):
        return f"{pid} {elapsed} /home/asg/.conda/envs/asgard/bin/{name}"

    def _write_log(self, prefix, age=0.5, started=None, data=True):
        started = self.now - 60 if started is None else started
        day = datetime.datetime.fromtimestamp(
            started, datetime.timezone.utc
        ).strftime("%Y%m%d")
        stamp = datetime.datetime.fromtimestamp(
            started + 1, datetime.timezone.utc
        ).strftime("%Y%m%dT%H%M%S")
        directory = self.root / day
        directory.mkdir(exist_ok=True)
        path = directory / f"{prefix}{stamp}.log"
        header = "# timestamp data\n" if prefix.startswith("ft_") else "time tx\n"
        path.write_text(header + ("1700000000.0 1.0\n" if data else ""))
        os.utime(path, (self.now - age, self.now - age))
        return path

    def _camera(self, ages=None, fps=1000, save_mode=1):
        ages = ages or {"baldr1": 1.0, "hei_k1": 1.0}
        return {
            "fps": fps,
            "save_mode": save_mode,
            "last_saved_unix_s": {
                name: self.now - age for name, age in ages.items()
            },
        }

    def test_all_fresh_and_partial_tt(self):
        self.ps_output = "\n".join(
            [self._writer("save-ft-performance"), self._writer("save-tt-performance", 101)]
        )
        self._write_log("ft_performance_")
        for beam in range(1, 5):
            self._write_log(f"btt_performance_beam{beam}_", age=3 if beam == 2 else 0.5)
        status = self.monitor.collect(self._camera())
        self.assertEqual(status["cred1"]["state"], "green")
        self.assertEqual(status["ft_performance"]["state"], "green")
        self.assertEqual(status["tt_performance"]["state"], "yellow")
        self.assertEqual(status["tt_performance"]["checks"]["beam2"]["state"], "stale")

    def test_slow_camera_uses_longer_limit_and_saving_off_is_red(self):
        camera = self._camera({"baldr1": 10.5, "hei_k1": 11.5}, fps=500)
        group = self.monitor.collect(camera)["cred1"]
        self.assertEqual(group["state"], "yellow")
        self.assertEqual(group["checks"]["baldr1"]["limit_s"], 11.0)
        camera["save_mode"] = 0
        self.assertEqual(self.monitor.collect(camera)["cred1"]["state"], "red")

    def test_camera_status_wrapper_from_zmq_is_decoded(self):
        response = json.dumps({"status_code": 0, "data": self._camera()})
        self.assertEqual(self.monitor.collect(response)["cred1"]["state"], "green")

    def test_missing_writer_and_unavailable_camera_are_red(self):
        status = self.monitor.collect(None)
        self.assertTrue(all(group["state"] == "red" for group in status.values()))
        self.assertEqual(
            status["ft_performance"]["checks"]["ft_performance"]["detail"],
            "writer not running",
        )

    def test_header_only_log_is_not_a_write(self):
        self.ps_output = self._writer("save-ft-performance")
        self._write_log("ft_performance_", data=False)
        check = self.monitor.collect(self._camera())["ft_performance"]["checks"][
            "ft_performance"
        ]
        self.assertEqual(check["state"], "stale")
        self.assertEqual(check["detail"], "no data rows")

    def test_concurrent_write_just_after_check_start_is_fresh(self):
        self.ps_output = self._writer("save-ft-performance")
        self._write_log("ft_performance_", age=-0.1)
        self.assertEqual(
            self.monitor.collect(self._camera())["ft_performance"]["state"],
            "green",
        )

    def test_restart_discovers_new_file(self):
        self.ps_output = self._writer("save-ft-performance", pid=100)
        self._write_log("ft_performance_", age=10)
        self.assertEqual(self.monitor.collect(self._camera())["ft_performance"]["state"], "red")
        self.ps_output = self._writer("save-ft-performance", pid=200, elapsed=5)
        stamp = datetime.datetime.fromtimestamp(
            self.now - 4, datetime.timezone.utc
        ).strftime("%Y%m%dT%H%M%S")
        day = stamp[:8]
        path = self.root / day / f"ft_performance_{stamp}.log"
        path.write_text("# timestamp data\n1700000000.0 1.0\n")
        os.utime(path, (self.now - 0.5, self.now - 0.5))
        self.assertEqual(
            self.monitor.collect(self._camera())["ft_performance"]["state"],
            "green",
        )

    def test_restart_does_not_credit_previous_process_write(self):
        self.ps_output = self._writer("save-ft-performance", pid=100)
        self._write_log("ft_performance_", age=0.5)
        self.assertEqual(
            self.monitor.collect(self._camera())["ft_performance"]["state"],
            "green",
        )
        self.ps_output = self._writer("save-ft-performance", pid=200, elapsed=5)
        self.assertEqual(
            self.monitor.collect(self._camera())["ft_performance"]["state"],
            "red",
        )

    def test_mcs_disk_status_request_is_separate_from_status(self):
        class Reply:
            payload = None

            def send_payload(self, payload, log_payload=True):
                self.payload = payload

        server = MCSServer.__new__(MCSServer)
        server.z = Reply()
        server.watchdog = type(
            "FakeWatchdog", (), {"collect_disk_status": lambda self: {"cred1": "ok"}}
        )()
        server.data = {"old": "value"}
        server.handle_message("disk_status")
        self.assertEqual(server.z.payload, {"cred1": "ok"})
        self.assertEqual(server.data, {})

    def test_log_from_previous_utc_day_remains_visible(self):
        midnight = datetime.datetime(2026, 10, 6, tzinfo=datetime.timezone.utc)
        self.now = midnight.timestamp() + 30
        self.ps_output = self._writer("save-ft-performance", elapsed=120)
        self._write_log("ft_performance_", started=self.now - 120)
        self.assertEqual(
            self.monitor.collect(self._camera())["ft_performance"]["state"],
            "green",
        )


if __name__ == "__main__":
    unittest.main()
