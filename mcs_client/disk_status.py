"""Check telemetry writes on mimir without reading /data from wag."""

import datetime
import heapq
import math
import os
import pathlib
import re
import shlex
import subprocess
import time

SOURCES = {
    "ft_performance": {
        "writer": "save-ft-performance",
        "prefixes": ("ft_performance_",),
        "labels": ("ft_performance",),
        "limit_s": 2.0,
    },
    "ft_settings": {
        "writer": "save-ft-performance",
        "prefixes": ("ft_settings_",),
        "labels": ("ft_settings",),
        "limit_s": 3.0,
    },
    "tt_performance": {
        "writer": "save-tt-performance",
        "prefixes": tuple(f"btt_performance_beam{beam}_" for beam in range(1, 5)),
        "labels": tuple(f"beam{beam}" for beam in range(1, 5)),
        "limit_s": 2.0,
    },
    "tt_settings": {
        "writer": "save-tt-performance",
        "prefixes": tuple(f"btt_beam{beam}_settings_" for beam in range(1, 5)),
        "labels": tuple(f"beam{beam}" for beam in range(1, 5)),
        "limit_s": 3.0,
    },
}
CRED1_STREAMS = tuple(f"baldr{beam}" for beam in range(1, 5)) + (
    "hei_k1",
    "hei_k2",
)
CRED1_WRITE_LIMIT_S = 10.0
DM_STREAM = "DM commands"
DM_WRITE_LIMIT_S = 10.0
DM_FILENAME = re.compile(r"dms_T\d{2}:\d{2}:\d{2}\.fits\Z")
DM_RECENT_FILE_CANDIDATES = 16


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
        self._cred1_cache = {}
        self._dm_cache = {}

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
                                    file_started = (
                                        datetime.datetime.strptime(
                                            stamp, "%Y%m%dT%H%M%S"
                                        )
                                        .replace(tzinfo=datetime.timezone.utc)
                                        .timestamp()
                                    )
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
            for _ in range(16):
                row = stream.readline(4096)
                if not row:
                    return None
                if not row.endswith(b"\n"):
                    return None
                data = row.split(None, 1)
                if len(data) < 2:
                    continue
                try:
                    if math.isfinite(float(data[0])):
                        return stat.st_mtime
                except ValueError:
                    continue
        return None

    def _log_group(self, source, writer, now):
        prefixes = SOURCES[source]["prefixes"]
        labels = SOURCES[source]["labels"]
        limit_s = SOURCES[source]["limit_s"]

        if writer is None:
            return _group(
                {
                    label: _check(None, limit_s, now, "writer not running")
                    for label in labels
                }
            )
        try:
            paths = self._paths_for(source, writer, now)
        except OSError as error:
            return _group(
                {label: _check(None, limit_s, now, str(error)) for label in labels}
            )
        checks = {}
        for prefix, name in zip(prefixes, labels):
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

    def _cred1_file_times(self, now):
        day = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).strftime(
            "%Y%m%d"
        )
        directory = self.data_root / day
        try:
            directory_mtime = directory.stat().st_mtime_ns
        except FileNotFoundError:
            return {}
        key = (directory, directory_mtime)
        if self._cred1_cache.get("key") != key:
            paths = {}
            with os.scandir(directory) as entries:
                for entry in entries:
                    name = entry.name
                    stream, separator, _ = name.partition("_T")
                    if (
                        not separator
                        or stream not in CRED1_STREAMS
                        or not name.endswith(".fits")
                    ):
                        continue
                    previous = paths.get(stream)
                    if previous is None or name > previous.name:
                        paths[stream] = pathlib.Path(entry.path)
            self._cred1_cache = {"key": key, "paths": paths}
        saved = {}
        for stream, path in self._cred1_cache["paths"].items():
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            if stat.st_size > 0:
                saved[stream] = stat.st_mtime
        return saved

    def _cred1_group(self, now):
        try:
            saved = self._cred1_file_times(now)
        except OSError as error:
            return _group(
                {
                    name: _check(None, CRED1_WRITE_LIMIT_S, now, str(error))
                    for name in CRED1_STREAMS
                }
            )
        checks = {
            name: _check(saved.get(name), CRED1_WRITE_LIMIT_S, now)
            for name in CRED1_STREAMS
        }
        return _group(checks)

    def _dm_file_time(self, now, started):
        today = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()
        latest = None
        for day in (today - datetime.timedelta(days=1), today):
            directory = self.data_root / day.strftime("%Y%m%d")
            try:
                directory_mtime = directory.stat().st_mtime_ns
            except FileNotFoundError:
                self._dm_cache.pop(directory, None)
                continue

            cached = self._dm_cache.get(directory)
            if cached is None or cached["mtime"] != directory_mtime:
                with os.scandir(directory) as entries:
                    names = heapq.nlargest(
                        DM_RECENT_FILE_CANDIDATES,
                        (entry.name for entry in entries if DM_FILENAME.fullmatch(entry.name)),
                    )
                cached = {
                    "mtime": directory_mtime,
                    "paths": [directory / name for name in names],
                }
                self._dm_cache[directory] = cached

            for path in cached["paths"]:
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    self._dm_cache.pop(directory, None)
                    continue
                if stat.st_size > 0 and started <= stat.st_mtime <= now + 1:
                    latest = max(latest, stat.st_mtime) if latest is not None else stat.st_mtime
        return latest

    def _dm_group(self, now, save_status):
        mode = save_status.get("mode") if isinstance(save_status, dict) else None
        started = (
            save_status.get("started_unix_s")
            if isinstance(save_status, dict)
            else None
        )
        if type(mode) is not int or mode not in (0, 1) or (
            mode == 1
            and (
                not isinstance(started, (int, float))
                or not math.isfinite(started)
                or started <= 0
                or started > now + 1
            )
        ):
            detail = "save mode unavailable"
            return {
                "mode": None,
                "detail": detail,
                **_group({DM_STREAM: _check(None, DM_WRITE_LIMIT_S, now, detail)}),
            }
        if mode == 0:
            return {
                "mode": 0,
                **_group(
                    {DM_STREAM: _check(None, DM_WRITE_LIMIT_S, now, "save mode off")}
                ),
            }
        try:
            saved = self._dm_file_time(now, started)
        except OSError as error:
            return {
                "mode": None,
                "detail": str(error),
                **_group(
                    {DM_STREAM: _check(None, DM_WRITE_LIMIT_S, now, str(error))}
                ),
            }
        detail = "" if saved is not None else "no nonempty DM FITS file since save 1"
        return {
            "mode": 1,
            **_group({DM_STREAM: _check(saved, DM_WRITE_LIMIT_S, now, detail)}),
        }

    def collect(self, dm_save_status=None):
        """Return source summaries and per-stream write ages for the ZMQ reply."""
        now = self.clock()
        try:
            writers = self._writers(self.process_list(), now)
        except (OSError, subprocess.CalledProcessError):
            writers = {}
        return {
            "cred1": self._cred1_group(now),
            "dm": self._dm_group(now, dm_save_status),
            **{
                source: self._log_group(source, writers.get(source), now)
                for source in SOURCES
            },
        }
