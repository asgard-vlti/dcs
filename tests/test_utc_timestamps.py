"""Offline checks for UTC service logs and backward-compatible DM flat names."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import datetime
import importlib
import io
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as onp

from mcs_client import mcs_client


INSTANT = datetime.datetime(2026, 10, 12, 1, 2, 3, 125000, datetime.timezone.utc)


class FixedDateTime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(INSTANT.timestamp(), tz)


@contextmanager
def host_timezone(name):
    try:
        with patch.dict(os.environ, {"TZ": name}):
            time.tzset()
            yield
    finally:
        time.tzset()


def log_record():
    record = logging.LogRecord("test", logging.WARNING, __file__, 1, "value=%s", (7,), None)
    record.created = INSTANT.timestamp()
    record.msecs = 125
    return record


class UTCLoggingTests(unittest.TestCase):
    def test_mcs_file_and_console_use_utc_independent_of_host_timezone(self):
        for zone in ("UTC", "America/Santiago"):
            with self.subTest(zone=zone), host_timezone(zone), tempfile.TemporaryDirectory() as directory:
                logger = logging.Logger("isolated-mcs")
                console = io.StringIO()
                with patch.object(logging, "root", logger), redirect_stderr(console), patch.object(
                    mcs_client, "datetime", FixedDateTime
                ), patch.object(mcs_client, "acquire_process_lock"), patch.object(
                    mcs_client, "MCSClient"
                ) as client, patch.object(sys, "argv", ["mcs-client", "--log-location", directory]):
                    mcs_client.main()
                    client.return_value.run.assert_called_once_with()
                    self.assertEqual(len(logger.handlers), 2)
                    self.assertEqual(logger.level, logging.INFO)
                    logger.handle(log_record())
                    for handler in logger.handlers:
                        handler.close()
                expected = "2026-10-12T01:02:03.125Z WARNING value=7\n"
                paths = list(Path(directory).iterdir())
                self.assertEqual([path.name for path in paths], ["20261012T010203Z.log"])
                self.assertEqual(paths[0].read_text(), expected)
                self.assertEqual(console.getvalue(), expected)

    def test_backend_console_uses_utc_without_starting_server(self):
        for zone in ("UTC", "America/Santiago"):
            with self.subTest(zone=zone), host_timezone(zone):
                logger = logging.Logger("isolated-backend")
                console = io.StringIO()
                with patch.object(logging, "root", logger), redirect_stderr(console), patch(
                    "zmq.Context", side_effect=AssertionError("No sockets in offline tests")
                ):
                    backend = importlib.import_module("back_end_server.back_end_server")
                    backend._setup_logging()
                    console.seek(0)
                    console.truncate()
                    logger.handle(log_record())
                    for handler in logger.handlers:
                        handler.close()
                self.assertEqual(
                    console.getvalue(), "2026-10-12T01:02:03.125Z WARNING: value=7\n"
                )


class FlatTimestampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared_memory = ModuleType("xaosim.shmlib")
        shared_memory.shm = Mock(side_effect=AssertionError("No shared memory in offline tests"))
        with patch.dict(sys.modules, {"xaosim.shmlib": shared_memory}):
            cls.save = importlib.import_module("dcs.cmd_scripts.flat_save")
            cls.load = importlib.import_module("dcs.cmd_scripts.flat_load")

    def test_save_uses_utc_and_preserves_arrays_and_archiving(self):
        data = onp.arange(144, dtype=onp.float64).reshape(12, 12)
        old_data = data + 100
        for zone in ("UTC", "America/Santiago"):
            with self.subTest(zone=zone), host_timezone(zone), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                beam = root / "beam2"
                beam.mkdir()
                old = beam / "night-standard_20261011_220203.npy"
                other = beam / "lab_20261011_220203.npy"
                onp.save(old, old_data)
                onp.save(other, old_data)
                with patch.object(self.save, "base_pth", root), patch.object(
                    self.save.datetime, "datetime", FixedDateTime
                ), patch.object(self.save, "read_dm_total", return_value=data) as read, redirect_stdout(io.StringIO()):
                    self.save.save_flat_for_beam(2, "night-standard")
                read.assert_called_once_with(2)
                output = beam / "night-standard_20261012T010203Z.npy"
                onp.testing.assert_array_equal(onp.load(output), data)
                self.assertEqual(onp.load(output).dtype, data.dtype)
                onp.testing.assert_array_equal(onp.load(beam / "archive" / old.name), old_data)
                self.assertFalse(old.exists())
                self.assertTrue(other.exists())

    def test_load_discovers_legacy_and_utc_names_without_reinterpreting_dates(self):
        data = onp.arange(144, dtype=onp.float64).reshape(12, 12)
        for name in ("lab_20261011_220203.npy", "lab_20261012T010203Z.npy"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                beam = root / "beam1"
                beam.mkdir()
                onp.save(beam / name, data)
                with patch.object(self.load, "base_pth", root), redirect_stdout(io.StringIO()):
                    onp.testing.assert_array_equal(self.load.load_flat(1, "lab"), data)
                    with patch.object(sys, "argv", ["flat-load", "1", "lab"]), patch.object(
                        self.load, "command_dm", return_value=True
                    ) as command:
                        self.load.main()
                    self.assertEqual(command.call_args.args[0], 1)
                    onp.testing.assert_array_equal(command.call_args.args[1], data)


if __name__ == "__main__":
    unittest.main()
