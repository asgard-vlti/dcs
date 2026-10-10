import datetime
import os
import pathlib
import tempfile
import unittest

from mcs_client.disk_status import CRED1_STREAMS, CRED1_WRITE_LIMIT_S, DiskSavingMonitor
from mcs_client.mcs_client import MCSServer, Watchdog


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

    def test_all_fresh_and_partial_tt(self):
        self.ps_output = "\n".join(
            [self._writer("save-ft-performance"), self._writer("save-tt-performance", 101)]
        )
        self._write_log("ft_performance_")
        self._write_log("ft_settings_")
        for beam in range(1, 5):
            self._write_log(f"btt_performance_beam{beam}_", age=3 if beam == 2 else 0.5)
            self._write_log(f"btt_beam{beam}_settings_")
        status = self.monitor.collect()
        self.assertEqual(status["cred1"]["state"], "red")
        self.assertEqual(status["ft_performance"]["state"], "green")
        self.assertEqual(status["ft_settings"]["state"], "green")
        self.assertEqual(status["tt_performance"]["state"], "yellow")
        self.assertEqual(status["tt_performance"]["checks"]["beam2"]["state"], "stale")
        self.assertEqual(status["tt_settings"]["state"], "green")

    def test_settings_freshness_and_missing_beam(self):
        self.ps_output = "\n".join(
            [self._writer("save-ft-performance"), self._writer("save-tt-performance", 101)]
        )
        self._write_log("ft_settings_", age=2.9)
        for beam in range(1, 4):
            self._write_log(f"btt_beam{beam}_settings_", age=3.0 if beam == 2 else 0.5)
        status = self.monitor.collect()
        self.assertEqual(status["ft_settings"]["state"], "green")
        self.assertEqual(status["ft_settings"]["checks"]["ft_settings"]["limit_s"], 3.0)
        self.assertEqual(status["tt_settings"]["state"], "yellow")
        self.assertEqual(status["tt_settings"]["checks"]["beam2"]["state"], "stale")
        self.assertEqual(
            status["tt_settings"]["checks"]["beam4"]["detail"], "log file missing"
        )
        self.now += 0.1
        self.assertEqual(self.monitor.collect()["ft_settings"]["state"], "red")

    def test_settings_header_only_log_and_restart(self):
        self.ps_output = self._writer("save-ft-performance")
        old_log = self._write_log("ft_settings_", data=False)
        check = self.monitor.collect()["ft_settings"]["checks"]["ft_settings"]
        self.assertEqual(check["state"], "stale")
        self.assertEqual(check["detail"], "no data rows")

        with old_log.open("a") as stream:
            stream.write("1700000000.0 1.0\n")
        os.utime(old_log, (self.now - 0.5, self.now - 0.5))
        self.assertEqual(self.monitor.collect()["ft_settings"]["state"], "green")

        self.ps_output = self._writer("save-ft-performance", pid=200, elapsed=5)
        check = self.monitor.collect()["ft_settings"]["checks"]["ft_settings"]
        self.assertEqual(check["state"], "stale")
        self.assertEqual(check["detail"], "log file missing")
        self.now += 5.1
        self._write_log("ft_settings_", started=self.now - 5)
        self.assertEqual(self.monitor.collect()["ft_settings"]["state"], "green")

    def test_camera_files_control_status_and_age_out(self):
        self.assertEqual(self.monitor.collect()["cred1"]["state"], "red")
        day = datetime.datetime.fromtimestamp(
            self.now, datetime.timezone.utc
        ).strftime("%Y%m%d")
        directory = self.root / day
        directory.mkdir()
        for stream in CRED1_STREAMS:
            path = directory / f"{stream}_T12:00:00.000.fits"
            path.write_bytes(b"FITS data")
            os.utime(path, (self.now - 0.5, self.now - 0.5))
            if stream == "baldr1":
                self.assertEqual(
                    self.monitor.collect()["cred1"]["state"], "yellow"
                )
        self.assertEqual(self.monitor.collect()["cred1"]["state"], "green")
        self.now += CRED1_WRITE_LIMIT_S
        self.assertEqual(self.monitor.collect()["cred1"]["state"], "red")

    def test_missing_writer_and_camera_files_are_red(self):
        status = self.monitor.collect()
        self.assertTrue(all(group["state"] == "red" for group in status.values()))
        self.assertEqual(
            status["ft_performance"]["checks"]["ft_performance"]["detail"],
            "writer not running",
        )
        self.assertEqual(
            status["ft_settings"]["checks"]["ft_settings"]["detail"],
            "writer not running",
        )
        self.assertEqual(
            status["tt_settings"]["checks"]["beam1"]["detail"],
            "writer not running",
        )

    def test_header_only_log_is_not_a_write(self):
        self.ps_output = self._writer("save-ft-performance")
        self._write_log("ft_performance_", data=False)
        check = self.monitor.collect()["ft_performance"]["checks"][
            "ft_performance"
        ]
        self.assertEqual(check["state"], "stale")
        self.assertEqual(check["detail"], "no data rows")

    def test_tt_log_updates_after_commit_headers_and_first_data_row(self):
        self.ps_output = self._writer("save-tt-performance")
        logs = [
            self._write_log(f"btt_performance_beam{beam}_", data=False)
            for beam in range(1, 5)
        ]
        for path in logs:
            path.write_text(
                "# dcs commit: abc\n"
                "# asgard-alignment commit: def\n"
                "time tx ty mx my\n"
            )
        self.assertEqual(
            self.monitor.collect()["tt_performance"]["checks"]["beam1"]["detail"],
            "no data rows",
        )
        for path in logs:
            with path.open("a") as stream:
                stream.write("1700000000.0 1.0 2.0 3.0 4.0\n")
            os.utime(path, (self.now - 0.5, self.now - 0.5))
        self.assertEqual(
            self.monitor.collect()["tt_performance"]["state"],
            "green",
        )

    def test_concurrent_write_just_after_check_start_is_fresh(self):
        self.ps_output = self._writer("save-ft-performance")
        self._write_log("ft_performance_", age=-0.1)
        self.assertEqual(
            self.monitor.collect()["ft_performance"]["state"],
            "green",
        )

    def test_restart_discovers_new_file(self):
        self.ps_output = self._writer("save-ft-performance", pid=100)
        self._write_log("ft_performance_", age=10)
        self.assertEqual(self.monitor.collect()["ft_performance"]["state"], "red")
        self.ps_output = self._writer("save-ft-performance", pid=200, elapsed=5)
        stamp = datetime.datetime.fromtimestamp(
            self.now - 4, datetime.timezone.utc
        ).strftime("%Y%m%dT%H%M%S")
        day = stamp[:8]
        path = self.root / day / f"ft_performance_{stamp}.log"
        path.write_text("# timestamp data\n1700000000.0 1.0\n")
        os.utime(path, (self.now - 0.5, self.now - 0.5))
        self.assertEqual(
            self.monitor.collect()["ft_performance"]["state"],
            "green",
        )

    def test_restart_does_not_credit_previous_process_write(self):
        self.ps_output = self._writer("save-ft-performance", pid=100)
        self._write_log("ft_performance_", age=0.5)
        self.assertEqual(
            self.monitor.collect()["ft_performance"]["state"],
            "green",
        )
        self.ps_output = self._writer("save-ft-performance", pid=200, elapsed=5)
        self.assertEqual(
            self.monitor.collect()["ft_performance"]["state"],
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

    def test_disk_status_does_not_query_camera(self):
        watchdog = Watchdog.__new__(Watchdog)
        watchdog.disk_monitor = type(
            "FakeMonitor", (), {"collect": lambda self: {"cred1": "files"}}
        )()
        self.assertEqual(watchdog.collect_disk_status(), {"cred1": "files"})

    def test_log_from_previous_utc_day_remains_visible(self):
        midnight = datetime.datetime(2026, 10, 6, tzinfo=datetime.timezone.utc)
        self.now = midnight.timestamp() + 30
        self.ps_output = self._writer("save-ft-performance", elapsed=120)
        self._write_log("ft_performance_", started=self.now - 120)
        self.assertEqual(
            self.monitor.collect()["ft_performance"]["state"],
            "green",
        )


if __name__ == "__main__":
    unittest.main()
