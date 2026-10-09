"""Check telemetry writes on mimir without reading /data from wag."""

import datetime
import json
import math
import os
import pathlib
import shlex
import subprocess
import time


SOURCES = {
    "ft_performance": {
        "writer": "save-ft-performance",
        "prefixes": ("ft_performance_",),
        "limit_s": 2.0,
    },
    "tt_performance": {
        "writer": "save-tt-performance",
        "prefixes": tuple(f"btt_performance_beam{beam}_" for beam in range(1, 5)),
        "limit_s": 2.0,
    },
}


def _check(last_saved, limit_s, now, detail=""):
    try:
        timestamp = float(last_saved)
    except (TypeError, ValueError):
        timestamp = 0.0
    age_s = now - timestamp if timestamp > 0 and math.isfinite(timestamp) else None
    if age_s is not None and -1.0 <= age_s < 0:
        age_s = 0.0
    if age_s is not None and age_s < -1.0:
        detail = "write timestamp is in the future"
    fresh = age_s is not None and 0 <= age_s < limit_s and not detail
    return {
        "state": "fresh" if fresh else "stale",
        "age_s": age_s,
        "limit_s": limit_s,
        "last_saved_unix_s": timestamp if age_s is not None else None,
        "detail": detail or ("" if fresh else "no recent write"),
    }


def _group(checks):
    fresh = sum(check["state"] == "fresh" for check in checks.values())
    return {
        "state": "green" if fresh == len(checks) else "yellow" if fresh else "red",
        "checks": checks,
    }


class DiskSavingMonitor:
    def __init__(self, data_root="/data", process_list=None, clock=None):
        self.data_root = pathlib.Path(data_root)
        self.process_list = process_list or self._process_list
        self.clock = clock or time.time
        self._file_cache = {}

    @staticmethod
    def _process_list():
        result = subprocess.run(
            ["ps", "-eo", "pid=,etimes=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    @staticmethod
    def _writers(ps_output, now):
        writers = {}
        for line in ps_output.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) != 3:
                continue
            pid, elapsed, command = parts
            try:
                argv = shlex.split(command)
                executable = pathlib.Path(argv[0]).name
                if executable.startswith("python") and len(argv) > 1:
                    executable = pathlib.Path(argv[1]).name
                started = now - int(elapsed)
            except (ValueError, IndexError):
                continue
            for source, spec in SOURCES.items():
                if executable == spec["writer"]:
                    writers[source] = (pid, started)
        return writers

    def _discover(self, started, prefixes):
        launch_day = datetime.datetime.fromtimestamp(
            started, datetime.timezone.utc
        ).date()
        candidates = {}
        for day in (launch_day, launch_day + datetime.timedelta(days=1)):
            directory = self.data_root / day.strftime("%Y%m%d")
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if not entry.name.endswith(".log"):
                            continue
                        for prefix in prefixes:
                            if entry.name.startswith(prefix):
                                stamp = entry.name[len(prefix) : len(prefix) + 15]
                                try:
                                    file_started = datetime.datetime.strptime(
                                        stamp, "%Y%m%dT%H%M%S"
                                    ).replace(tzinfo=datetime.timezone.utc).timestamp()
                                except ValueError:
                                    continue
                                if file_started < started - 2:
                                    continue
                                previous = candidates.get(prefix)
                                if previous is None or entry.name > previous.name:
                                    candidates[prefix] = pathlib.Path(entry.path)
            except FileNotFoundError:
                continue
        return candidates

    def _paths_for(self, source, writer, now):
        pid, started = writer
        prefixes = SOURCES[source]["prefixes"]
        key = (
            pid,
            datetime.datetime.fromtimestamp(started, datetime.timezone.utc).date(),
        )
        cache = self._file_cache.get(source)
        if cache is None or cache["key"] != key:
            cache = {"key": key, "paths": {}, "last_discovery": 0.0}
            self._file_cache[source] = cache
        if len(cache["paths"]) < len(prefixes) and now - cache["last_discovery"] >= 5:
            cache["paths"] = self._discover(started, prefixes)
            cache["last_discovery"] = now
        return cache["paths"]

    @staticmethod
    def _file_timestamp(path):
        stat = path.stat()
        if stat.st_size == 0:
            return None
        with path.open("rb") as stream:
            stream.readline(4096)
            row = stream.readline(4096)
        if not row.endswith(b"\n"):
            return None
        data = row.split(None, 1)
        if len(data) < 2:
            return None
        try:
            if not math.isfinite(float(data[0])):
                return None
        except ValueError:
            return None
        return stat.st_mtime

    def _log_group(self, source, writer, now):
        prefixes = SOURCES[source]["prefixes"]
        limit_s = SOURCES[source]["limit_s"]

        def label(prefix):
            return prefix.replace("btt_performance_", "").rstrip("_")

        if writer is None:
            return _group({
                label(prefix): _check(None, limit_s, now, "writer not running")
                for prefix in prefixes
            })
        try:
            paths = self._paths_for(source, writer, now)
        except OSError as error:
            return _group({
                label(prefix): _check(None, limit_s, now, str(error))
                for prefix in prefixes
            })
        checks = {}
        for prefix in prefixes:
            name = label(prefix)
            path = paths.get(prefix)
            if path is None:
                checks[name] = _check(None, limit_s, now, "log file missing")
                continue
            try:
                saved_at = self._file_timestamp(path)
                checks[name] = _check(
                    saved_at, limit_s, now, "no data rows" if saved_at is None else ""
                )
            except OSError as error:
                checks[name] = _check(None, limit_s, now, str(error))
        return _group(checks)

    @staticmethod
    def _cred1_group(camera_status, now):
        for _ in range(2):
            if isinstance(camera_status, str):
                try:
                    camera_status = json.loads(camera_status)
                except ValueError:
                    camera_status = None
            if isinstance(camera_status, dict) and "data" in camera_status:
                camera_status = camera_status["data"]
        if not isinstance(camera_status, dict):
            return _group({"CRED1": _check(None, 6.0, now, "cannot verify camera")})
        saved = camera_status.get("last_saved_unix_s")
        if not isinstance(saved, dict) or not saved:
            return _group({"CRED1": _check(None, 6.0, now, "save times unavailable")})
        try:
            fps = float(camera_status.get("fps"))
            limit_s = max(6.0, 5000.0 / fps + 1.0) if fps > 0 else 6.0
        except (TypeError, ValueError, ZeroDivisionError):
            limit_s = 6.0
        if not math.isfinite(limit_s):
            limit_s = 6.0
        saving_off = camera_status.get("save_mode") != 1
        checks = {
            name: _check(timestamp, limit_s, now, "saving off" if saving_off else "")
            for name, timestamp in saved.items()
        }
        return _group(checks)

    def collect(self, camera_status):
        """Return source summaries and per-stream write ages for the ZMQ reply."""
        now = self.clock()
        try:
            writers = self._writers(self.process_list(), now)
        except (OSError, subprocess.CalledProcessError):
            writers = {}
        return {
            "cred1": self._cred1_group(camera_status, now),
            **{
                source: self._log_group(source, writers.get(source), now)
                for source in SOURCES
            },
        }
