"""Acquire one standard-mode Baldr beam for later optical-model fitting.

Run from this directory with::

    python measure_baldr_standard_for_model.py 1

The CRED1 server's existing dark subtraction is left untouched. Every saved image,
including the shuttered dark cube, comes from the same processed camera stream.
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as onp
from tqdm.auto import tqdm


OUTPUT_DIR = Path("~/tmp/AT_baldr_fitting").expanduser()
MOTOR_NAMES = ("BMX", "BMY", "BTP", "BTT")
POKE_ACTUATORS = ((3, 3), (3, 8), (8, 3), (8, 8))
SHUTTER_BMY_UM = 500.0


@dataclass(frozen=True)
class AcquisitionConfig:
    dark_frames: int = 256
    pupil_frames: int = 256
    reference_frames: int = 1000
    probe_frames: int = 2
    probe_repeats: int = 10
    discard_frames: int = 1
    probe_amplitude: float = 0.03
    dm_settle_s: float = 0.01
    motor_settle_s: float = 2.0
    motor_poll_s: float = 0.05
    motor_timeout_s: float = 60.0
    mask_offset_um: float = -200.0
    dither_deg: float = 0.03
    # A short subset is useful for offline tests; production uses every mode.
    mode_count: Optional[int] = None


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MDSClient:
    """Use the RTC's MDS helpers, reconnecting after a timed-out request."""

    def __init__(self, host="mimir", port=5555):
        from minimal_baldr_python_rtc import utils

        self.utils = utils
        self.host = host
        self.port = port
        self.context = None
        self.socket = None
        self._connect()

    def _connect(self):
        import zmq

        self.context, self.socket = self.utils.mds_connect(
            self.host, self.port, timeout_ms=10000
        )
        self.socket.setsockopt(zmq.SNDTIMEO, 10000)
        self.socket.setsockopt(zmq.LINGER, 0)

    def send(self, command):
        import zmq

        try:
            response = self.utils.mds_send(self.socket, command)
        except zmq.ZMQError:
            self.close()
            self._connect()
            raise
        if response.upper().startswith(("NACK", "ERROR")):
            raise RuntimeError(f"MDS rejected {command!r}: {response}")
        return response

    def close(self):
        if self.socket is not None:
            self.socket.close(0)
            self.socket = None
        if self.context is not None:
            self.context.term()
            self.context = None


def query_service_status(port):
    """Read a local instrument service status without changing its settings."""
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.SNDTIMEO, 1000)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect(f"tcp://mimir:{port}")
    try:
        socket.send_string("status")
        reply = socket.recv_string().strip("\x00\n ")
        try:
            return json.loads(reply)
        except json.JSONDecodeError:
            return {"raw": reply}
    except zmq.ZMQError:
        return None
    finally:
        socket.close(0)
        context.term()


def read_positions(mds, beam):
    return {
        name: float(mds.send(f"read {name}{beam}")) for name in MOTOR_NAMES
    }


def wait_for_position(mds, beam, name, target, config):
    axis = f"{name}{beam}"
    tolerance = 2.0 if name in ("BMX", "BMY") else 0.002
    deadline = time.monotonic() + config.motor_timeout_s
    consecutive = 0
    while time.monotonic() < deadline:
        actual = float(mds.send(f"read {axis}"))
        consecutive = consecutive + 1 if abs(actual - target) <= tolerance else 0
        if consecutive >= 2:
            time.sleep(config.motor_settle_s)
            return actual
        time.sleep(config.motor_poll_s)
    raise TimeoutError(f"{axis} did not reach {target}; last position {actual}")


def move_motor(mds, beam, name, target, config):
    mds.send(f"moveabs {name}{beam} {target:.6f}")
    return wait_for_position(mds, beam, name, target, config)


def take_frames(cam, count):
    # False bypasses Cam's additional dark correction, not the CRED1 server's.
    frames = onp.asarray(cam.take_stack(count, subtract_dark=False), dtype=onp.float32)
    if frames.ndim != 3 or frames.shape[0] != count:
        raise ValueError(f"Camera returned {frames.shape}, expected {count} frames")
    return frames


def set_dm_delta(dm, starting_channel, delta):
    # Channel 0 holds the flat; only the RTC's experiment channel is written.
    command = starting_channel + delta
    dm.shms[2].set_data(command)
    dm.shm0.post_sems(1)
    return command


def record_stage(record, stage, mds, beam):
    record["stage_positions"][stage] = {
        "utc": utc_now(),
        "positions": read_positions(mds, beam),
    }


def acquire_probes(record, key, dm, cam, starting_channel, delta_maps, config):
    with tqdm(
        total=len(delta_maps) * config.probe_repeats * 2,
        desc=key.replace("_", " "),
        unit="probe",
        ascii=True,
        disable=not sys.stderr.isatty(),
    ) as progress:
        for mode_index, positive_delta in enumerate(delta_maps):
            for repeat in range(config.probe_repeats):
                for sign in (-1, 1):
                    record["dm_touched"] = True
                    command = set_dm_delta(dm, starting_channel, sign * positive_delta)
                    time.sleep(config.dm_settle_s)
                    if config.discard_frames:
                        take_frames(cam, config.discard_frames)
                    record[f"{key}_frames"].append(take_frames(cam, config.probe_frames))
                    record[f"{key}_dm_channel2_commands"].append(command.copy())
                    record[f"{key}_index"].append(mode_index)
                    record[f"{key}_repeat"].append(repeat)
                    record[f"{key}_sign"].append(sign)
                    record[f"{key}_utc"].append(utc_now())
                    progress.update(1)
    set_dm_delta(dm, starting_channel, onp.zeros_like(starting_channel))


def acquire_sequence(record, beam, dm, cam, mds, config):
    starting = read_positions(mds, beam)
    record["starting_positions"] = starting
    if "SHUTTERED" in mds.send(f"state BMY{beam}").upper():
        raise RuntimeError(f"BMY{beam} is already shuttered")

    mask_target = starting["BMY"] + config.mask_offset_um
    if not 0 <= mask_target <= 10000:
        raise ValueError(f"BMY target {mask_target} µm is outside [0, 10000]")
    offsets = (-config.dither_deg, 0.0, config.dither_deg)
    for name in ("BTP", "BTT"):
        if not all(-0.75 <= starting[name] + offset <= 0.75 for offset in offsets):
            raise ValueError(f"{name}{beam} dither would exceed ±0.75 degrees")

    basis = onp.asarray(dm.basis, dtype=onp.float64)
    if basis.ndim != 2 or basis.shape[0] != 144:
        raise ValueError(f"Expected a 144-actuator Fourier basis, got {basis.shape}")
    if len(dm.shms) <= 2 or dm.main_chn != 2:
        raise RuntimeError("DM channel 2 is unavailable")
    mode_count = basis.shape[1] if config.mode_count is None else config.mode_count
    if not 1 <= mode_count <= basis.shape[1]:
        raise ValueError("Invalid Fourier mode count")

    record["fourier_basis"] = basis.astype(onp.float32)
    record["starting_dm_channel2"] = onp.asarray(
        dm.shms[2].get_data(), dtype=onp.float64
    ).reshape(-1).copy()
    record["starting_dm_combined"] = onp.asarray(
        dm.shm0.get_data(), dtype=onp.float32
    ).reshape(-1).copy()
    if record["starting_dm_channel2"].size != 144:
        raise ValueError("DM channel 2 does not contain 144 actuators")

    fourier_deltas = config.probe_amplitude * basis[:, :mode_count]
    fourier_deltas -= fourier_deltas.mean(axis=0, keepdims=True)
    fourier_deltas = fourier_deltas.T.copy()
    poke_deltas = onp.zeros((len(POKE_ACTUATORS), 144), dtype=onp.float64)
    for index, (row, column) in enumerate(POKE_ACTUATORS):
        poke_deltas[index, row * 12 + column] = config.probe_amplitude
        poke_deltas[index] -= poke_deltas[index].mean()
    record["fourier_delta_maps"] = fourier_deltas.astype(onp.float32)
    record["poke_delta_maps"] = poke_deltas.astype(onp.float32)

    # Set the flag before sending: a timed-out close may still have reached MDS.
    record["motors_touched"] = True
    record["shutter_may_be_closed"] = True
    mds.send(f"b_shut close {beam}")
    wait_for_position(mds, beam, "BMY", SHUTTER_BMY_UM, config)
    record_stage(record, "dark", mds, beam)
    record["dark_utc"] = utc_now()
    record["dark_frames"] = take_frames(cam, config.dark_frames)

    mds.send(f"b_shut open {beam}")
    wait_for_position(mds, beam, "BMY", starting["BMY"], config)
    record["shutter_may_be_closed"] = False

    move_motor(mds, beam, "BMY", mask_target, config)
    record_stage(record, "pupil_start", mds, beam)
    with tqdm(
        total=len(offsets) ** 2,
        desc="pupil grid",
        unit="position",
        ascii=True,
        disable=not sys.stderr.isatty(),
    ) as progress:
        for pitch_offset, tilt_offset in product(offsets, offsets):
            move_motor(mds, beam, "BTP", starting["BTP"] + pitch_offset, config)
            move_motor(mds, beam, "BTT", starting["BTT"] + tilt_offset, config)
            positions = read_positions(mds, beam)
            captured_at = utc_now()
            frames = take_frames(cam, config.pupil_frames)
            record["pupil_dither_offsets"].append((pitch_offset, tilt_offset))
            record["pupil_dither_positions"].append(positions)
            record["pupil_dither_utc"].append(captured_at)
            record["pupil_dither_frames"].append(frames)
            progress.update(1)
    move_motor(mds, beam, "BTP", starting["BTP"], config)
    move_motor(mds, beam, "BTT", starting["BTT"], config)

    record_stage(record, "pupil_fourier_start", mds, beam)
    acquire_probes(
        record,
        "pupil_fourier",
        dm,
        cam,
        record["starting_dm_channel2"],
        fourier_deltas,
        config,
    )
    record_stage(record, "pupil_fourier_end", mds, beam)

    move_motor(mds, beam, "BMY", starting["BMY"], config)
    record_stage(record, "masked_reference", mds, beam)
    record["masked_reference_utc"] = utc_now()
    record["masked_reference_frames"] = take_frames(cam, config.reference_frames)

    record_stage(record, "masked_fourier_start", mds, beam)
    acquire_probes(
        record,
        "masked_fourier",
        dm,
        cam,
        record["starting_dm_channel2"],
        fourier_deltas,
        config,
    )
    record_stage(record, "masked_fourier_end", mds, beam)

    record_stage(record, "masked_poke_start", mds, beam)
    acquire_probes(
        record,
        "masked_poke",
        dm,
        cam,
        record["starting_dm_channel2"],
        poke_deltas,
        config,
    )
    record_stage(record, "masked_poke_end", mds, beam)


def restore_state(record, beam, dm, mds, config):
    errors = []
    starting_channel = record.get("starting_dm_channel2")
    if starting_channel is not None and record["dm_touched"]:
        try:
            set_dm_delta(dm, starting_channel, onp.zeros_like(starting_channel))
        except Exception as exc:
            errors.append(f"DM channel 2: {exc}")

    starting = record.get("starting_positions")
    if starting is None:
        return errors
    if not record["motors_touched"]:
        try:
            record["final_positions"] = read_positions(mds, beam)
        except Exception as exc:
            errors.append(f"final motor read: {exc}")
        return errors
    if record["shutter_may_be_closed"]:
        try:
            mds.send(f"b_shut open {beam}")
            wait_for_position(mds, beam, "BMY", starting["BMY"], config)
            record["shutter_may_be_closed"] = False
        except Exception as exc:
            errors.append(f"Baldr shutter: {exc}")

    for name in ("BTP", "BTT", "BMY"):
        try:
            move_motor(mds, beam, name, starting[name], config)
        except Exception as exc:
            errors.append(f"{name}{beam}: {exc}")
    try:
        record["final_positions"] = read_positions(mds, beam)
    except Exception as exc:
        errors.append(f"final motor read: {exc}")
    return errors


def packed_frames(frames, image_shape, frames_per_sample):
    if isinstance(frames, list):
        if frames:
            return onp.stack(frames)
        return onp.empty((0, frames_per_sample, *image_shape), dtype=onp.float32)
    return frames


def save_record(path, record, beam, cam, config, failure, cleanup_errors):
    image_shape = tuple(cam.imsize)
    metadata = {
        "schema_version": 2,
        "beam": beam,
        "created_utc": record["created_utc"],
        "config": asdict(config),
        "camera_stream": f"/dev/shm/baldr{beam}.im.shm",
        "camera_server_dark_subtracted": True,
        "frame_treatment": "server output; no additional dark subtraction or normalization",
        "analysis_dark_use": "subtract the mean of dark_frames from measurement frames",
        "motor_units": {"BMX": "um", "BMY": "um", "BTP": "deg", "BTT": "deg"},
        "motor_axis_order": MOTOR_NAMES,
        "probe_sign_order": (-1, 1),
        "poke_actuator_order": POKE_ACTUATORS,
        "dm_command_layout": "Each *_dm_channel2_commands row is the full 144-actuator channel 2 command sent for the matching *_frames stack; no DM readback.",
        "starting_positions": record.get("starting_positions"),
        "stage_positions": record["stage_positions"],
        "final_positions": record.get("final_positions"),
        "dark_utc": record.get("dark_utc"),
        "masked_reference_utc": record.get("masked_reference_utc"),
        "camera_status": record.get("camera_status"),
        "rtc_status": record.get("rtc_status"),
        "complete": failure is None and not cleanup_errors,
        "failure": None if failure is None else f"{type(failure).__name__}: {failure}",
        "cleanup_errors": cleanup_errors,
    }
    arrays = {
        "metadata_json": onp.array(json.dumps(metadata)),
        "beam": onp.array(beam),
        "complete": onp.array(metadata["complete"]),
        "dark_frames": record.get(
            "dark_frames", onp.empty((0, *image_shape), dtype=onp.float32)
        ),
        "pupil_dither_frames": packed_frames(
            record["pupil_dither_frames"], image_shape, config.pupil_frames
        ),
        "pupil_dither_offsets_deg": onp.asarray(
            record["pupil_dither_offsets"], dtype=onp.float64
        ).reshape(-1, 2),
        "pupil_dither_positions": onp.asarray(
            [[positions[name] for name in MOTOR_NAMES] for positions in record["pupil_dither_positions"]],
            dtype=onp.float64,
        ).reshape(-1, len(MOTOR_NAMES)),
        "pupil_dither_utc": onp.asarray(record["pupil_dither_utc"], dtype=str),
        "masked_reference_frames": record.get(
            "masked_reference_frames",
            onp.empty((0, *image_shape), dtype=onp.float32),
        ),
        "fourier_basis": record.get("fourier_basis", onp.empty((144, 0))),
        "fourier_delta_maps": record.get("fourier_delta_maps", onp.empty((0, 144))),
        "poke_delta_maps": record.get("poke_delta_maps", onp.empty((0, 144))),
        "poke_actuators": onp.asarray(POKE_ACTUATORS, dtype=onp.int16),
        "starting_dm_channel2": record.get("starting_dm_channel2", onp.empty(0)),
        "starting_dm_combined": record.get("starting_dm_combined", onp.empty(0)),
    }
    for key in ("pupil_fourier", "masked_fourier", "masked_poke"):
        arrays[f"{key}_frames"] = packed_frames(
            record[f"{key}_frames"], image_shape, config.probe_frames
        )
        arrays[f"{key}_dm_channel2_commands"] = onp.asarray(
            record[f"{key}_dm_channel2_commands"], dtype=onp.float64
        ).reshape(-1, 144)
        for label, dtype in (("index", onp.int16), ("repeat", onp.int16), ("sign", onp.int8)):
            arrays[f"{key}_{label}"] = onp.asarray(record[f"{key}_{label}"], dtype=dtype)
        arrays[f"{key}_utc"] = onp.asarray(record[f"{key}_utc"], dtype=str)

    temporary_path = path.with_name(path.stem + ".partial.npz")
    try:
        onp.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run_and_save(beam, dm, cam, mds, output_dir=OUTPUT_DIR, config=None, rtc_status=None, camera_status=None):
    config = config or AcquisitionConfig()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = output_dir / f"beam{beam}_{timestamp}.npz"
    record = {
        "created_utc": utc_now(),
        "rtc_status": rtc_status,
        "camera_status": camera_status,
        "stage_positions": {},
        "shutter_may_be_closed": False,
        "motors_touched": False,
        "dm_touched": False,
        "pupil_dither_frames": [],
        "pupil_dither_offsets": [],
        "pupil_dither_positions": [],
        "pupil_dither_utc": [],
    }
    for key in ("pupil_fourier", "masked_fourier", "masked_poke"):
        for label in ("frames", "dm_channel2_commands", "index", "repeat", "sign", "utc"):
            record[f"{key}_{label}"] = []

    failure = None
    try:
        if isinstance(rtc_status, dict) and (
            rtc_status.get("servo") == "on" or rtc_status.get("wants_to_close")
        ):
            raise RuntimeError("Baldr RTC servo is active or waiting to close")
        acquire_sequence(record, beam, dm, cam, mds, config)
    except BaseException as exc:
        failure = exc

    cleanup_errors = restore_state(record, beam, dm, mds, config)
    save_record(path, record, beam, cam, config, failure, cleanup_errors)
    print(f"Saved {'complete' if failure is None and not cleanup_errors else 'partial'} data to {path}")
    if cleanup_errors:
        raise RuntimeError("State restoration failed: " + "; ".join(cleanup_errors)) from failure
    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Acquire standard-mode Baldr frames for model fitting."
    )
    parser.add_argument("beam", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()

    from minimal_baldr_python_rtc import Cam, DM

    dm = DM.FourierDM(args.beam)
    cam = Cam.Cam(args.beam)
    mds = MDSClient()

    def on_sigterm(_signum, _frame):
        raise InterruptedError("Received SIGTERM")

    previous_sigterm = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        rtc_status = query_service_status(6661 + args.beam)
        camera_status = query_service_status(6667)
        run_and_save(
            args.beam, dm, cam, mds, rtc_status=rtc_status, camera_status=camera_status
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        mds.close()


if __name__ == "__main__":
    main()
