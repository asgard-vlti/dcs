"""Power-cycle the C-RED1 camera and restore its cropped server mode on mimir."""

import fcntl
import json
import shutil
import subprocess
import sys
import time

import zmq


CAMERA_ENDPOINT = "tcp://127.0.0.1:6667"
CAMERA_LOCK = "/tmp/asg.cam_server.lock"
PDU_HOST = "192.168.100.11"
PDU_OUTLET = 6
KAYA_HOST = "192.168.100.10"

REQUEST_TIMEOUT_MS = 10000
SERVER_STOP_TIMEOUT_S = 30
SERVER_START_TIMEOUT_S = 60
PDU_TIMEOUT_S = 30
POWER_OFF_S = 5
COOLING_TIMEOUT_S = 15 * 60
POLL_INTERVAL_S = 5


def send_command(command):
    """Make one bounded REQ exchange, using a fresh socket after each request."""
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.setsockopt(zmq.SNDTIMEO, REQUEST_TIMEOUT_MS)
    socket.setsockopt(zmq.RCVTIMEO, REQUEST_TIMEOUT_MS)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(CAMERA_ENDPOINT)
        socket.send_string(command)
        raw = socket.recv_string()
    except zmq.ZMQError as exc:
        raise RuntimeError(f"cam_server did not answer {command!r}: {exc}") from exc
    finally:
        socket.close()

    try:
        response = json.loads(raw)
    except json.JSONDecodeError:
        response = raw
    if isinstance(response, str) and response.lstrip().lower().startswith("error:"):
        raise RuntimeError(f"cam_server rejected {command!r}: {response.strip()}")
    print(f"cam_server {command}: {response}", flush=True)
    return response


def send_cli(command):
    return send_command(f"cli {json.dumps(command)}")


def wait_for_server_stop():
    """Wait for the process lock to be released; the lock file itself persists."""
    deadline = time.monotonic() + SERVER_STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with open(CAMERA_LOCK, "r") as lock_file:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    return
        except FileNotFoundError:
            return
        time.sleep(0.5)
    raise RuntimeError("cam_server did not release its lock after exit")


def stop_server():
    response = send_command("exit")
    if str(response).strip() != "Exiting!":
        raise RuntimeError(f"Unexpected cam_server exit response: {response!r}")
    wait_for_server_stop()


def start_server():
    subprocess.run(
        ["run_cam_server"],
        check=True,
        timeout=15,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    last_error = None
    while time.monotonic() < deadline:
        try:
            status = send_command("status")
            if isinstance(status, dict) and "cam_status" in status:
                return
            last_error = f"invalid server status: {status!r}"
        except RuntimeError as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(
        f"cam_server did not start within {SERVER_START_TIMEOUT_S} seconds: {last_error}"
    )


def wait_for_outlet(pdu, expected):
    deadline = time.monotonic() + PDU_TIMEOUT_S
    last_status = None
    while time.monotonic() < deadline:
        last_status = pdu.read_outlet_status(PDU_OUTLET)
        if last_status == expected:
            print(f"PDU outlet {PDU_OUTLET}: {expected}", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(
        f"PDU outlet {PDU_OUTLET} did not become {expected}; last status: {last_status!r}"
    )


def cycle_pdu(pdu):
    needs_power_on = True
    try:
        pdu.switch_outlet_status(PDU_OUTLET, "off")
        wait_for_outlet(pdu, "off")
        time.sleep(POWER_OFF_S)
        pdu.switch_outlet_status(PDU_OUTLET, "on")
        wait_for_outlet(pdu, "on")
        needs_power_on = False
    finally:
        if needs_power_on:
            try:
                pdu.switch_outlet_status(PDU_OUTLET, "on")
                wait_for_outlet(pdu, "on")
            except Exception as exc:
                print(f"ERROR: Could not restore PDU outlet {PDU_OUTLET}: {exc}", file=sys.stderr)


def cli_state_is(response, state, label):
    for line in str(response).splitlines():
        value = line.strip().strip("> ").strip().lower()
        if value == state:
            return True
        if ":" in value:
            key, value = value.split(":", 1)
            if key.strip() == label and value.strip() == state:
                return True
    return False


def monitor_camera(deadline, kaya_class):
    last_temp = None
    last_status = None
    kaya_restarted = False
    while True:
        try:
            last_temp = send_command("get_det_temp")
        except RuntimeError as exc:
            last_temp = str(exc)
            print(f"Temperature query failed: {exc}", file=sys.stderr)
        try:
            last_status = send_cli("status")
        except RuntimeError as exc:
            last_status = str(exc)
            print(f"Camera status query failed: {exc}", file=sys.stderr)
        status_observed_at = time.monotonic()
        if not kaya_restarted:
            restart_kaya(kaya_class)
            kaya_restarted = True
        if status_observed_at <= deadline and cli_state_is(
            last_status, "operational", "status"
        ):
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "Camera did not become operational within 15 minutes; "
                f"last temperature: {last_temp!r}; last status: {last_status!r}"
            )
        time.sleep(min(POLL_INTERVAL_S, remaining))


def restart_kaya(kaya_class):
    subprocess.run(["restart-kaya"], check=True, timeout=45)
    kaya = kaya_class(KAYA_HOST, init_motors=False)
    try:
        status = kaya.get_status("Kaya")
        if status.strip() != "1":
            raise RuntimeError(f"Kaya did not report powered on: {status!r}")
        print("Kaya reports powered on", flush=True)
    finally:
        if kaya.client is not None:
            kaya.disconnect()


def preflight():
    for command in ("run_cam_server", "restart-kaya"):
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} is not on PATH")
    from asgard_alignment.PDU_telnet import AtenEcoPDU
    from asgard_alignment.controllino import PowerControllino

    return AtenEcoPDU, PowerControllino


def power_cycle_camera():
    pdu_class, kaya_class = preflight()
    pdu = pdu_class(PDU_HOST)
    try:
        if not pdu.connect():
            raise RuntimeError(f"Could not connect to PDU at {PDU_HOST}")
        send_cli("set cooling off")
        send_cli("shutdown")
        stop_server()
        cycle_pdu(pdu)
    finally:
        pdu.close()

    start_server()
    send_cli("set cooling on")
    cooling_deadline = time.monotonic() + COOLING_TIMEOUT_S
    monitor_camera(cooling_deadline, kaya_class)
    stop_server()
    start_server()
    cropping = send_cli("cropping")
    if not cli_state_is(cropping, "active", "cropping"):
        raise RuntimeError(f"Camera cropping is not active: {cropping!r}")
    print("Camera operational; cam_server restarted with cropped mode", flush=True)


def main():
    try:
        power_cycle_camera()
    except KeyboardInterrupt:
        print("Interrupted; camera power was restored if outlet 6 was off", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, subprocess.SubprocessError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
