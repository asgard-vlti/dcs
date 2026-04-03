"""
Connect to the socket and save a fixed number of tip/tilt metrology points
to a fits file (number as an optional argv input)

Data will come in as 4 json lists of variable length:
tx, ty, mx, my

The "cnt" variable is returned each time.
"""

import ZMQ_control_client as zmq_client
import json
import numpy as np
from astropy.io import fits
import sys
import time
import pathlib

from dcs.ZMQutils import ZmqReq

ports = [
    6671,
    6672,
    6673,
    6674,
]

# reqs = [
#     ZmqReq("tcp://
# ]


def get_zmq(port):
    while True:
        try:
            return ZmqReq(f"tcp://192.168.100.2:{port}")
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
                f.write(" ".join(self.FIELDS) + "\n")

    def log_performance(self):
        h_z = get_zmq(self.zmq_port)

        # get the data since last cnt
        data = h_z.send_payload(f"ttmet {self.last_cnt}", is_str=True, decode_ascii=False)
        if type(data) != dict:
            raise UserWarning(f"Data unexpected: {data}")

        # log to file
        with open(self.log_path, "a") as f:
            for i in range(len(data["tx"])):
                values = [data[field][i] for field in self.FIELDS]
                f.write(" ".join(f"{v:.3f}" for v in values) + "\n")

        # Update last cnt
        self.last_cnt = data["cnt"]


if __name__ == "__main__":

    cur_datetime = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    year_month_day = time.strftime("%Y%m%d", time.gmtime())
    pth = f"/data/{year_month_day}"
    pathlib.Path(pth).mkdir(parents=True, exist_ok=True)

    loggers = []

    for beam_idx in range(4):
        log_path = f"{pth}/btt_performance_beam{beam_idx}_{cur_datetime}.log"
        logger = BTTLogger(log_path, ports[beam_idx], last_cnt=0)
        loggers.append(logger)
        print(f"Logging BTT performance for beam {beam_idx} to {log_path}")
    while True:
        try:
            for logger in loggers:
                logger.log_performance()
        except Exception as e:
            print(
                f"[BTT Performance] Error logging performance: {e}. Retrying in 2s..."
            )
            time.sleep(2)

    # if len(sys.argv) > 1:
    #     n_points = int(sys.argv[1])
    # else:
    #     n_points = 4096

    # print(f"Saving {n_points} tip/tilt metrology points to fits file.")
    # tx_list = []
    # ty_list = []
    # mx_list = []
    # my_list = []
    # cnt = 0
    # while len(tx_list) < n_points:
    #     data = zmq_client.send(f"ttmet {cnt}")
    #     if type(data) != dict:
    #         raise UserWarning("Incorrect response: " + data)
    #     tx_list += data["tx"]
    #     ty_list += data["ty"]
    #     mx_list += data["mx"]
    #     my_list += data["my"]
    #     cnt = data["cnt"]
    #     print(f"Got point {len(tx_list):04d}/{n_points}, cnt={cnt}", end="\r")
    #     time.sleep(0.01)
