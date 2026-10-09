"""
A script to save the FT performance, namely:
- The complex value at the centre of each splodge, from Frantz's code
- The dm pistons and dl offloads from Heimdallr

Logs to a text file, running indefinitely until interrupted.
"""

import time
import argparse
import os
import sys
import fcntl
from dcs.ZMQutils import ZmqReq
from dcs.log_commit_ids import commit_header

import threading

LOCK_FILE_PATH = "/tmp/asg.heim_telem.lock"

keys_of_interest = [
    "gd_snr",
    "pd_snr",
    "gd_bl",
    "pd_tel",
    "gd_tel",
    "dm_piston",
]

# Settings keys and their order
settings_keys = [
    "servo_mode",
    "n_gd_boxcar",
    "gd_threshold",
    "pd_threshold",
    "gd_search_reset",
    "offload_time_ms",
    "offload_gd_gain",
    "gd_gain",
    "kp",
]


def log_ft_performance(
    h_z,
    initial_reply,
    log_path="ft_performance_log.txt",
):
    logger = FTPerformanceLogger(h_z, initial_reply, log_path)
    try:
        while True:
            t0 = time.monotonic()
            try:
                backlog = logger.poll_once()
            except ConnectionError:
                time.sleep(0.1)
                continue
            if backlog:
                continue
            interval = 0.1 if logger.servo_mode == 4 else 0.01
            time.sleep(max(0, interval - (time.monotonic() - t0)))
    finally:
        logger.close()


def wait_for_telemetry(h_z):
    while True:
        reply = h_z.send_payload("ft_telemetry 0,0,0", is_str=True, decode_ascii=False)
        if isinstance(reply, dict) and {
            "stream_id",
            "latest_seq",
            "dropped_total",
        } <= reply.keys():
            return reply
        status = h_z.send_payload("status", is_str=True, decode_ascii=False)
        if isinstance(status, dict) and "cnt" in status:
            raise RuntimeError(
                "Heimdallr is reachable but ft_telemetry is unavailable; deploy the updated server first"
            )
        time.sleep(1)


class FTPerformanceLogger:
    def __init__(self, h_z, initial_reply, log_path):
        self.h_z = h_z
        self.stream_id = initial_reply["stream_id"]
        self.last_seq = initial_reply["latest_seq"]
        self.dropped_total = initial_reply["dropped_total"]
        self.announced_drops = 0
        self.servo_mode = 4
        self.file = open(log_path, "a+")
        self.file.seek(0, os.SEEK_END)
        if self.file.tell() == 0:
            self.file.write(commit_header())
            self.file.write(
                "# timestamp gd_snr pd_snr gd_bl pd_tel gd_tel dm_piston cnt seq servo_mode "
                "(measurements: 3 decimal places; counters and mode: integers)\n"
            )
            self.file.flush()

    def close(self):
        self.file.close()

    def poll_once(self):
        reply = self.h_z.send_payload(
            f"ft_telemetry {self.stream_id},{self.last_seq},128",
            is_str=True,
            decode_ascii=False,
        )
        if reply is None:
            raise ConnectionError("No FT telemetry response")
        if not isinstance(reply, dict):
            raise RuntimeError(f"Unexpected FT telemetry response: {reply}")

        stream_id = reply["stream_id"]
        reset = stream_id != self.stream_id
        if reset != reply["reset"]:
            raise RuntimeError("FT telemetry stream reset flag disagrees with stream ID")
        rows = reply["rows"]
        lines = []
        last_seq = self.last_seq
        announced_drops = self.announced_drops
        dropped_total = reply["dropped_total"]
        servo_mode = self.servo_mode

        if reset:
            lines.append(f"# telemetry_restart stream_id={stream_id}\n")
            last_seq = None
            dropped_total = reply["dropped_total"]
            missing_since_restart = max(0, rows[0]["seq"] - 1) if rows else 0
            if missing_since_restart:
                lines.append(
                    f"# telemetry_loss since_restart={missing_since_restart}\n"
                )
            announced_drops = max(0, dropped_total - missing_since_restart)
            if announced_drops:
                lines.append(
                    f"# telemetry_loss producer_dropped={announced_drops}\n"
                )
        elif dropped_total < self.dropped_total:
            raise RuntimeError("FT telemetry drop counter went backwards")
        else:
            new_drops = dropped_total - self.dropped_total
            if new_drops:
                lines.append(f"# telemetry_loss producer_dropped={new_drops}\n")
                announced_drops += new_drops

        for row in rows:
            seq = row["seq"]
            if last_seq is not None:
                gap = (seq - last_seq - 1) & 0xFFFFFFFF
                if gap >= 0x80000000:
                    raise RuntimeError("FT telemetry rows are out of order")
                accounted = min(gap, announced_drops)
                announced_drops -= accounted
                if gap > accounted:
                    lines.append(
                        f"# telemetry_loss missing={gap - accounted} "
                        f"after_seq={last_seq} before_seq={seq}\n"
                    )
            rounded_time = (row["time_ns"] + 50_000) // 100_000
            timestamp = f"{rounded_time // 10_000}.{rounded_time % 10_000:04d}"
            values = []
            for key in keys_of_interest:
                values.extend(
                    f"{float(value) if value is not None else float('nan'):.3f}"
                    for value in row[key]
                )
            lines.append(
                f"{timestamp} {' '.join(values)} {row['cnt']} {seq} "
                f"{row['servo_mode']}\n"
            )
            last_seq = seq
            servo_mode = row["servo_mode"]

        if reset and not rows:
            last_seq = reply["latest_seq"]

        if lines:
            self.file.write("".join(lines))
            self.file.flush()
        self.stream_id = stream_id
        self.last_seq = last_seq
        self.dropped_total = dropped_total
        self.announced_drops = announced_drops
        self.servo_mode = servo_mode
        return bool(rows) and last_seq != reply["latest_seq"]


def log_ft_settings(h_z, log_path="ft_settings_log.txt", rate_hz=1):
    """
    Logs FT settings to a file at a slower rate (default 1 Hz).
    """
    write_header = True
    try:
        with open(log_path, "r") as f_check:
            if f_check.read(1):
                write_header = False
    except FileNotFoundError:
        pass
    with open(log_path, "a") as f:
        if write_header:
            f.write(commit_header())
            f.write(
                "# timestamp "
                + " ".join(settings_keys)
                + " (all values, 3 decimal places)\n"
            )
        while True:
            t0 = time.time()
            try:
                reply = h_z.send_payload(
                    "settings", is_str=True, decode_ascii=False
                )
            except Exception as e:
                print(f"[FT Settings] Error during request: {e}. Retrying...")
                time.sleep(1)
                continue
            if reply:
                timestamp = "{:.3f}".format(t0)
                values = []
                for k in settings_keys:
                    v = reply.get(k)
                    try:
                        values.append("{:.3f}".format(float(v)))
                    except Exception:
                        values.append(str(v))
                line = f"{timestamp} {' '.join(values)}"
                f.write(line + "\n")
                f.flush()
            time.sleep(max(0, (1.0 / rate_hz) - (time.time() - t0)))


def acquire_process_lock(lock_path=LOCK_FILE_PATH):
    """Acquire a non-blocking process lock and record current PID in the lock file."""
    lock_file = open(lock_path, "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(f"lock file is already locked: {lock_path}")

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def main():
    parser = argparse.ArgumentParser(
        description="Start logging the fringe tracker performance and settings."
    )
    parser.add_argument(
        "--gdrate",
        type=int,
        default=10,
        help="Legacy off-mode rate; only the default of 10 Hz is accepted",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=1000,
        help="Legacy active rate; only the default of 1000 Hz is accepted",
    )
    parser.add_argument(
        "--is-sim",
        action="store_true",
        help="Connect to the local simulator and save logs under sim-data",
    )
    args = parser.parse_args()
    if args.rate != 1000 or args.gdrate != 10:
        parser.error(
            "custom --rate/--gdrate values are unsupported: capture is fixed at every active frame and at most 10 Hz while off"
        )

    try:
        _instance_lock = acquire_process_lock()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # time in UTC
    cur_datetime = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    fname = f"ft_performance_{cur_datetime}.log"
    year_month_day = time.strftime("%Y%m%d", time.gmtime())
    data_dir = (
        "/home/taras/Documents/0projects/asgard/sim-data" if args.is_sim else "/data"
    )
    pth = os.path.join(data_dir, year_month_day)
    # Make directories if they don't exist
    os.makedirs(pth, exist_ok=True)
    full_pth = os.path.join(pth, fname)
    # Settings log file
    settings_fname = f"ft_settings_{cur_datetime}.log"
    settings_full_pth = os.path.join(pth, settings_fname)

    # single socket and lock
    endpoint = "tcp://127.0.0.1:6660" if args.is_sim else "tcp://192.168.100.2:6660"
    h_z = ZmqReq(endpoint)
    initial_reply = wait_for_telemetry(h_z)
    settings_client = ZmqReq(endpoint)

    settings_thread = threading.Thread(
        target=log_ft_settings,
        args=(settings_client, settings_full_pth, 1),
        daemon=True,
    )
    settings_thread.start()
    try:
        log_ft_performance(h_z, initial_reply, full_pth)
    except KeyboardInterrupt:
        print("Logging stopped.")


# Example usage:
if __name__ == "__main__":
    main()
