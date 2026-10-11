#!/usr/bin/env python3
"""Measure Heimdallr pupil drift and correct it through MDS."""

import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as onp
import zmq


ddir = "/data/custom/hmd_pupil/"
default_log = ddir + "log_pupil_monitor.log"
mds_address = "tcp://192.168.100.2:5555"
pupil_error_threshold = 0.05


def log(message="", logfile=default_log, echo=True):
    tstamp = datetime.utcnow().strftime("%D %H:%M:%S")
    myline = f"{tstamp}: {message}"

    with open(logfile, "a") as mylog:
        mylog.write(myline + "\n")
    if echo:
        print(myline)


def average_visibility(hmd, dstream, nim=50, im_offset=1000):
    v2 = []
    for _ in range(nim):
        img = dstream.get_data() - im_offset
        cvis = hmd.get_raw_cvis(hmd.dense, img, full=True)
        v2.append(onp.abs(cvis) ** 2)
    return onp.mean(v2, axis=0)


def pupil_move_commands(errors):
    errors = onp.asarray(errors)
    if errors.shape != (4, 2) or not onp.isfinite(errors).all():
        raise ValueError("Pupil errors must be four finite (x, y) pairs")

    commands = []
    for beam, (x_error, y_error) in enumerate(errors, start=1):
        if onp.hypot(x_error, y_error) > pupil_error_threshold:
            commands.append(
                (beam, f"mv_pup c_red_one_focus {beam} {-x_error:.6f} {-y_error:.6f}")
            )
    return commands


def send_pupil_moves(commands, dry_run=False):
    if not commands:
        print("No pupil moves required.")
        return

    if dry_run:
        for _, command in commands:
            print(f"DRY RUN: {command}")
        return

    context = zmq.Context()
    socket = None
    moved_beams = []
    try:
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        socket.setsockopt(zmq.SNDTIMEO, 10000)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(mds_address)
        for beam, command in commands:
            try:
                socket.send_string(command)
                response = socket.recv_string().strip()
            except zmq.ZMQError as exc:
                raise RuntimeError(
                    f"MDS failed on beam {beam}; confirmed moved beams: "
                    f"{moved_beams}; beam {beam} may have moved"
                ) from exc
            if response != "ACK: moved":
                raise RuntimeError(
                    f"MDS replied {response!r} on beam {beam}; "
                    f"confirmed moved beams: {moved_beams}"
                )
            moved_beams.append(beam)
            print(f"{command} -> {response}")
    finally:
        if socket is not None:
            socket.close()
        context.term()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Show MDS commands without moving pupils"
    )
    args = parser.parse_args(argv)

    try:
        from hmd_tts import HMD_TTS
    except ImportError:
        from pyeng_heimdallr.tip_tilt_sensor.hmd_tts import HMD_TTS
    from xaosim.shmlib import shm

    hmd = HMD_TTS()
    dstream = shm("/dev/shm/hei_k1.im.shm")
    v2 = average_visibility(hmd, dstream)

    xy0 = hmd.sparse.kpi.VAC[:, :2]
    xy1 = hmd.infer_pupil_model(v2)
    errors = onp.asarray(xy1) - onp.asarray(xy0)
    commands = pupil_move_commands(errors)

    uvc = hmd.dense.kpi.UVC
    uu = onp.append(uvc[:, 0], -uvc[:, 0])
    vv = onp.append(uvc[:, 1], -uvc[:, 1])
    tstamp = datetime.utcnow().strftime("%Y%m%d_T%H:%M:%S_")

    f1, ax = plt.subplots()
    ax.scatter(uu, vv, c=v2, vmax=0.3)
    ax.grid(True)
    f1.set_size_inches(5, 5, forward=True)
    f1.set_tight_layout(True)
    f1.savefig(f"{ddir}/pwsp/{tstamp}.png")
    plt.close(f1)

    f2, ax = plt.subplots()
    ax.scatter(xy0[:, 0], xy0[:, 1], c="b", s=14**2, label="Model")
    ax.scatter(xy1[:, 0], xy1[:, 1], c="r", s=14**2, label="Measured")
    for ii in range(4):
        ax.text(xy0[ii, 0] - 0.2, xy0[ii, 1] - 0.2, f"B{ii + 1}", fontsize=14)
    ax.legend(loc=1)
    ax.set_xlim([-1.7, 1.7])
    ax.set_ylim([-1.7, 1.7])
    ax.grid(True)
    f2.set_size_inches(5, 5, forward=True)
    f2.set_tight_layout(True)
    f2.savefig(f"{ddir}/pupil/{tstamp}.png")
    plt.close(f2)

    log(
        onp.array2string(
            errors.flatten(),
            suppress_small=True,
            floatmode="fixed",
            precision=3,
            separator=", ",
        )
    )
    send_pupil_moves(commands, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
