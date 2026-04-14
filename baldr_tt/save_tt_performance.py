"""
Connect to the socket and save a fixed number of tip/tilt metrology points
to a fits file (number as an optional argv input)

Data will come in as 4 json lists of variable length:
tx, ty, mx, my

The "cnt" variable is returned each time.
"""

import numpy as np
import time
import pathlib
import os
import sys
import fcntl

from dcs.ZMQutils import ZmqReq


LOCK_FILE_PATH = "/tmp/asg.baldr_tt_telem.lock"

ports = [
    6671,
    6672,
    6673,
    6674,
]

settings_to_log = [
    "flux_threshold",
    "focus_amp",
    "focus_offset",
    "gauss_hwidth",
    "hog",
    "hol",
    "px",
    "py",
    "servo_mode",
    "ttg",
    "ttl",
    "ttxo",
    "ttyo",
]


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


def get_zmq(port):
    while True:
        try:
            z = ZmqReq(f"tcp://192.168.100.2:{port}")
            print(f"[BTT Performance] Connected to server on port {port}")
            return z

        except Exception as e:
            print(
                f"[FT Performance] Could not connect to server: {e}. Retrying in 2s..."
            )
            time.sleep(2)


class BTTLogger:
    FIELDS = [
        "tx",
        "ty",
        "mx",
        "my",
    ]

    def __init__(self, log_path, zmq_port, last_cnt):
        self.log_path = log_path
        self.zmq_port = zmq_port
        self.last_cnt = last_cnt
        self.h_z = get_zmq(self.zmq_port)
        # Write header only if file is empty
        write_header = True
        try:
            with open(log_path, "r") as f_check:
                if f_check.read(1):
                    write_header = False
        except FileNotFoundError:
            pass

        if write_header:
            with open(log_path, "w") as f:
                f.write(" ".join(["time"] + self.FIELDS) + "\n")

    def log_performance(self):
        # get the data since last cnt

        try:
            data = self.h_z.send_payload(
                f"ttmet {self.last_cnt}", is_str=True, decode_ascii=False
            )
        except Exception as e:
            print(
                f"[FT Performance] Lost connection to server: {e}. Reconnecting in 2s..."
            )
            time.sleep(2)
            self.h_z = get_zmq(self.zmq_port)
            return

        if type(data) != dict:
            raise UserWarning(f"Data unexpected: {data}")

        # log to file
        with open(self.log_path, "a") as f:
            timestamp = "{:.4f}".format(time.time())
            for i in range(len(data["tx"])):
                if i == 0:
                    t = str(timestamp)
                else:
                    t = "prev"
                values = [data[field][i] for field in self.FIELDS]
                f.write(" ".join([t] + [f"{v:.3f}" for v in values]) + "\n")

        # Update last cnt
        self.last_cnt = data["cnt"]


class BTTSettingLogger:
    """
    logs much slower, doesnt need to keep track of last count
    """

    def __init__(self, log_path, zmq_port):
        self.log_path = log_path
        self.zmq_port = zmq_port
        self.h_z = get_zmq(self.zmq_port)
        # Write header only if file is empty
        write_header = True
        try:
            with open(log_path, "r") as f_check:
                if f_check.read(1):
                    write_header = False
        except FileNotFoundError:
            pass

        if write_header:
            with open(log_path, "w") as f:
                f.write("# timestamp " + " ".join(settings_to_log) + "\n")

    def log_settings(self):
        try:
            reply = self.h_z.send_payload("settings", is_str=True, decode_ascii=False)
        except Exception as e:
            print(
                f"[FT Performance] Lost connection to server: {e}. Reconnecting in 2s..."
            )
            time.sleep(2)
            self.h_z = get_zmq(self.zmq_port)
            return

        with open(self.log_path, "a") as f:
            timestamp = "{:.4f}".format(time.time())
            values = []
            for k in settings_to_log:
                v = reply.get(k)
                if isinstance(v, (list, np.ndarray)):
                    values.extend(
                        [
                            "{:.3f}".format(float(x) if x is not None else np.nan)
                            for x in v
                        ]
                    )
                else:
                    try:
                        values.append("{:.3f}".format(float(v)))
                    except Exception:
                        values.append(str(v))
            line = "{} {}".format(timestamp, " ".join(values))
            f.write(line + "\n")


def main():
    try:
        _instance_lock = acquire_process_lock()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    cur_datetime = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    year_month_day = time.strftime("%Y%m%d", time.gmtime())
    pth = f"/data/{year_month_day}"
    pathlib.Path(pth).mkdir(parents=True, exist_ok=True)

    loggers = []

    for beam_idx in range(4):
        log_path = f"{pth}/btt_performance_beam{beam_idx+1}_{cur_datetime}.log"
        logger = BTTLogger(log_path, ports[beam_idx], last_cnt=0)
        loggers.append(logger)
        print(f"Logging BTT performance for beam {beam_idx} to {log_path}")

    settings_loggers = []
    for beam_idx in range(4):
        settings_log_path = f"{pth}/btt_beam{beam_idx+1}_settings_{cur_datetime}.log"
        settings_logger = BTTSettingLogger(settings_log_path, ports[beam_idx])
        settings_loggers.append(settings_logger)
        print(f"Logging BTT settings for beam {beam_idx} to {settings_log_path}")

    last_settings_log_time = time.time()
    setttings_cadence = 1.0  # log settings every 1 second

    while True:
        try:
            for logger in loggers:
                logger.log_performance()

            if time.time() - last_settings_log_time > setttings_cadence:
                for settings_logger in settings_loggers:
                    settings_logger.log_settings()
                last_settings_log_time = time.time()

        except Exception as e:
            print(
                f"[BTT Performance] Error logging performance: {e}. Retrying in 2s..."
            )
            time.sleep(2)


if __name__ == "__main__":
    main()
