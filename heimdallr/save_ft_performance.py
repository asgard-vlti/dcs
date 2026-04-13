"""
A script to save the FT performance, namely:
- The complex value at the centre of each splodge, from Frantz's code
- The dm pistons and dl offloads from Heimdallr

Logs to a text file, running indefinitely until interrupted.
"""

import zmq
import json
import numpy as np
import time
import argparse
from typing import Optional
import os
from dcs.ZMQutils import ZmqReq

import threading

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
    lock,
    shared_state,
    log_path="ft_performance_log.txt",
    rate_hz=1000,
    gd_rate_hz=10,
):
    # Write header only if file is empty
    write_header = True
    try:
        with open(log_path, "r") as f_check:
            if f_check.read(1):
                write_header = False
    except FileNotFoundError:
        pass
    with open(log_path, "a") as f:
        if write_header:
            f.write(
                "# timestamp gd_snr pd_snr gd_bl pd_tel gd_tel dm_piston (all values space-separated, 3 decimal places)\n"
            )
        last_cnt = 0
        while True:
            t0 = time.time()
            try:
                with lock:
                    reply = h_z.send_payload("status", is_str=True, decode_ascii=False)
            except Exception as e:
                print(f"[FT Performance] Error during request: {e}. Retrying...")
                time.sleep(1)
                continue
            if reply:
                if "cnt" in reply:
                    cnt = reply["cnt"]
                    if last_cnt == cnt:
                        continue
                    last_cnt = cnt
                # Timestamp to ms precision
                timestamp = "{:.4f}".format(t0)
                # Flatten all key values into a single line, 3 decimal places
                values = []
                for k in keys_of_interest:
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
                f.flush()
            if (
                shared_state.get("servo_mode", 4) == 4
            ):  # If in "off" mode, we can log settings at a slower rate
                time.sleep(max(0, (1.0 / gd_rate_hz) - (time.time() - t0)))
            else:
                time.sleep(max(0, (1.0 / rate_hz) - (time.time() - t0)))


def log_ft_settings(h_z, lock, shared_state, log_path="ft_settings_log.txt", rate_hz=1):
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
            f.write(
                "# timestamp "
                + " ".join(settings_keys)
                + " (all values, 3 decimal places)\n"
            )
        while True:
            t0 = time.time()
            try:
                with lock:
                    reply = h_z.send_payload(
                        "settings", is_str=True, decode_ascii=False
                    )
            except Exception as e:
                print(f"[FT Settings] Error during request: {e}. Retrying...")
                time.sleep(1)
                continue
            if reply:
                # Adjust gd rate
                if "servo_mode" in reply:
                    shared_state["servo_mode"] = reply["servo_mode"]
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


def main():
    parser = argparse.ArgumentParser(description="FT performance data logging script")
    parser.add_argument(
        "--gdrate",
        type=int,
        default=10,
        help="Sample rate when group delay tracking in Hz",
    )
    parser.add_argument("--rate", type=int, default=1000, help="Sample rate in Hz")
    args = parser.parse_args()
    # time in UTC
    cur_datetime = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    fname = f"ft_performance_{cur_datetime}.log"
    year_month_day = time.strftime("%Y%m%d", time.gmtime())
    pth = f"/data/{year_month_day}"
    # Make directories if they don't exist
    os.makedirs(pth, exist_ok=True)
    full_pth = os.path.join(pth, fname)
    # Settings log file
    settings_fname = f"ft_settings_{cur_datetime}.log"
    settings_full_pth = os.path.join(pth, settings_fname)

    # single socket and lock
    h_z = ZmqReq("tcp://192.168.100.2:6660")
    lock = threading.Lock()
    shared_state = {"servo_mode": 4}

    # Start both logging functions in separate threads
    t1 = threading.Thread(
        target=log_ft_performance,
        args=(h_z, lock, shared_state, full_pth, args.rate, args.gdrate),
        daemon=True,
    )
    t2 = threading.Thread(
        target=log_ft_settings,
        args=(h_z, lock, shared_state, settings_full_pth, 1),
        daemon=True,
    )
    t1.start()
    t2.start()
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Logging stopped.")


# Example usage:
if __name__ == "__main__":
    main()
