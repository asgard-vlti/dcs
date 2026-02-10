"""
BackEndServer
=============

Purpose
- Exposes a single ZeroMQ REP endpoint to receive WAG/DCS JSON commands and return
  the standard reply shape: {"reply": {"time": "...", "content": "OK" | "ERROR: ..."}}.
- Implements core verbs (ping, setup, start, abort, expstatus) and dispatches
  RTS (real-time system) commands to subsystem-specific handlers via a registry
  (e.g., handlers.baldr). Baldr commands may be sent as "bld_<name>".

Why a bounded worker pool for RTS?
- A REP socket must strictly recv→send→recv…; if the command handler blocks,
  all clients are head-of-line blocked (even for different subsystems/ports).
- RTS commands are treated as *fire-and-forget*: we ACK immediately once accepted,
  then complete work in the background and push results to WAG/MCS via update_DB().
- We use a bounded ThreadPoolExecutor plus a semaphore cap to:
    * avoid unbounded thread creation and memory growth,
    * prevent the REP loop from blocking on long RTS actions,
    * provide predictable back-pressure ("ERROR: busy" when saturated).

Concurrency / safety notes
- Do not share ZMQ REQ sockets across threads. Concrete RTS tasks must
  create/connect any device sockets inside run() (per-thread) and close them.
- The main thread only parses, validates, enqueues, and immediately replies "OK".

Operational features
- Registry pattern (`register_rts`) selects the task class by command name
  (the "bld_" prefix is stripped). Adding a new RTS = add class + register once.
- `report_jobs` returns a snapshot of queued/running/done jobs (job_id, name, state, err).
- `abort` supports cancel-by-job_id; concrete tasks implement actual termination logic.
- When the pool is saturated, requests receive `"ERROR: busy"` rather than blocking.

This design keeps the wire contract simple, avoids REP head-of-line blocking, and
scales predictably while remaining easy to extend with new RTS handlers.
"""

""" 
SPECIFIC DETAILS FROM 

Asgard Top-Level Control Software User and Maintenance Manual

This back-end-server takes commands from wag, and based on a database of:
command, server, internal_command for checking what is possible:

This is the back_end_server.py

The server has to implement a nubmber of commands, and after each successful command, it should return:
{
“reply”:
{
“time” : “< timestamp >”
“content” : “OK”
,
}
}

Commands to be implemented:

PING: recieve:
{
“command” :
{
“name” : “ping”
,
“time” : “< timestamp >”
}
}


SETUP: recieve:
{
“command” :
{
“name” : “setup”
,
“time” : “< timestamp >”
“parameters” :
,
[
{
“name” : “< keyword #1 name >”
“value” : < keyword #1 value>
},
. . .
{
“name” : “< keyword #n name >”
“value” : < keyword #n value>
}
,
,
]
}

NB the setup command will be used in the asg_hdlr_only_autotest_align template, e.g. for now
we are using DCS readout, 1000Hz, and gain=10. DET.DIT=0.001, DET.NDIT=0 (i.e. no saving),
DET.NWORESET=2.

We need START (name:start) and ABORT (name:abort) commands, which are used to start and abort the readout.

Once we get to on-sky data, we need an EXPSTATUS command.
{
“command” :
{
“name” : “expstatus”
,
“time” : “< timestamp >”
}
}

The RTS command is used for RTS commands, which are lower case and generally passed directoy to servers.
{
command:
{
“name” : “< name of the command >”
,
“time” : “< timestamp >”
,
“parameters” :
[
{
“name” : “< name of parameter 1 >”
“value” : < value of parameter 1 >
,
},
. . .
{
“name” : “< name of parameter n >”
“value” : < value of parameter n >
,
}
]
}
}
e.g. "name" : "beamid", "value" : 1,2,3,4 or 0 (for all available)
"name: "server" : "heimdallr", "baldr", "cam"

If an error, then replace "OK" in the reply with “ERROR: < error description>”

Sockets:
cam_server      6667
DM_server       6666
hdlr            6660
hdlr_align      6661
"baldr1"        6662,
"baldr2"        6663,
"baldr3"        6664,
"baldr4"        6665,
"MCS"           7020,
"ICS"           5555,
"RTD"           7000,
"""

import zmq
import json
import datetime
import time
import os
import signal
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Type, Any
from pathlib import Path
import sys
import subprocess
import logging

# from rts_base import AbstractRTSTask, RTSContext, RTSState, RTSErr
from handlers.baldr_rts_handlers import register as register_baldr_rts


# --- Logging setup: file and console ---
def _setup_logging():
    log_dir = os.path.expanduser("~/logs/back_end_server/")
    os.makedirs(log_dir, exist_ok=True)
    log_name = time.strftime("back_end_server_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(log_dir, log_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging started. Log file: {log_path}")


_setup_logging()


# port 7004 is used by nomachines! changed to 7010
class BackEndServer:
    def __init__(
        self,
        port=7002,
        server_ports={
            "hdlr": 6660,
            "hdlr_align": 6661,
            "baldr1": 6662,
            "baldr2": 6663,
            "baldr3": 6664,
            "baldr4": 6665,
            "cam_server": 6667,
            "DM_server": 6666,
        },
    ):
        self.port = port
        # self.baldr_commands = baldr_back_end_server.BaldrCommands() #This is the Ben way.
        self.context = zmq.Context()

        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        logging.info(f"BackEndServer started on port {self.port}")

        self.server_ports = server_ports
        self.servers = {}
        for server_name, port in self.server_ports.items():
            server = self.context.socket(zmq.REQ)
            server.RCVTIMEO = 5000  # timeout in ms
            server.SNDTIMEO = 5000  # timeout in ms
            try:
                server.connect(f"tcp://mimir:{port}")
                self.servers[server_name] = server
                logging.info(f"Connected to {server_name} on port {port}")
            except zmq.ZMQError as e:
                logging.error(f"Failed to connect to {server_name} on port {port}: {e}")
                self.servers[server_name] = None
                # Handle connection error (e.g., retry, log, etc.)
                continue
        logging.info("All server connections initialized.")

        # Store total requested integration time
        self.itime = 0

        self.scripts_running = {}

    def run(self):
        check_scripts_every = 5  # seconds
        last_check = time.time()
        while True:
            # Wait for the next request from wag
            message = self.socket.recv().decode("ascii")
            if message[-1] == "\0":
                message = message[:-1]
            # Parse the message as JSON
            try:
                message = json.loads(message)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON: {e}")
                self.socket.send_json(
                    self.create_response("ERROR: Invalid JSON format")
                )
                continue
            logging.info(f"Received request: {message}")

            # Process the command
            response = self.process_command(message)

            logging.info(
                f"Scripts running after processing command: {self.scripts_running}"
            )

            # Periodically check the status of running scripts and remove from dict if done
            if time.time() - last_check > check_scripts_every:
                last_check = time.time()
                for pid, process in list(self.scripts_running.items()):
                    retcode = process.poll()
                    if retcode is not None:  # Process has finished
                        logging.info(
                            f"Script with PID {pid} finished with return code {retcode}."
                        )
                        stdout, stderr = process.communicate()
                        logging.info("Process output:")
                        logging.info(stdout)
                        if stderr:
                            logging.error("Process errors:")
                            logging.error(stderr)
                        del self.scripts_running[pid]

            # Send the reply back to wag
            logging.info(f"Sending: {response}")
            self.socket.send_json(response)

    def process_command(self, message):
        command = message.get("command", {})
        command_name = command.get("name", "").lower()

        if command_name == "ping":
            return self.create_response("OK")
        elif command_name == "setup":
            return self.setup(command)
        elif command_name == "start":
            return self.start(command)
        elif command_name == "abort":
            return self.abort()
        elif command_name == "expstatus":
            return self.expstatus(command)
        elif command_name == "foreground":
            return self.foreground(command)
        elif command_name.startswith("bld_"):
            # Fire-and-forget RTS
            return self.handle_bld_rts(command)
        elif command_name.startswith("hdlr_"):
            return self.handle_hdlr_rts(command)
        elif command_name.startswith("s_"):
            return self.handle_script(command)
        else:
            return self.create_response(f"ERROR: Unknown command '{command_name}'")

    def handle_bld_rts(self, command):
        """
        Dispatch BALDR RTS commands to the appropriate BALDR RTC Commander sockets.
        Self-contained; accepts RTS-format parameter list:
            command = {
            "name": "bld_open_lo" | "bld_open_ho" | "bld_close_lo" | "bld_close_ho",
            "time": "...",  # ignored
            "parameters": [
                {"name": "beam", "value": 0|1|2|3|4}
            ]
            }
        Semantics:
        - beam=0 => broadcast to beams 1..4
        - beam in 1..4 => target that specific beam
        """

        # Map WAG verbs -> Commander command strings
        cmd_map = {
            "bld_open_lo": 'open_baldr_LO ""',
            "bld_open_ho": 'open_baldr_HO ""',
            "bld_close_lo": 'close_baldr_LO ""',
            "bld_close_ho": 'close_baldr_HO ""',
            "bld_n0_update": 'N0_update ""',
        }

        name = (command.get("name") or "").lower()
        if name not in cmd_map:
            return self.create_response(f"ERROR: Unknown BALDR RTS command '{name}'")

        # ---- Extract single 'beam' from RTS-style parameters list ----
        params = command.get("parameters", [])

        def _param_value(params_obj, key, default=None):
            # Accepts the RTS list format and (optionally) a dict format
            if isinstance(params_obj, list):
                for item in params_obj:
                    if isinstance(item, dict) and item.get("name") == key:
                        return item.get("value", default)
                return default
            if isinstance(params_obj, dict):
                return params_obj.get(key, default)
            return default

        beam_raw = _param_value(params, "beam", 0)
        try:
            beam_val = int(beam_raw)
        except Exception:
            return self.create_response(
                "ERROR: 'beam' must be an integer in {0,1,2,3,4}"
            )
        if beam_val not in (0, 1, 2, 3, 4):
            return self.create_response("ERROR: 'beam' must be in {0,1,2,3,4}")

        # Expand 0 -> [1,2,3,4], otherwise a single-element list
        target_beams = [1, 2, 3, 4] if beam_val == 0 else [beam_val]

        # ---- send to each requested BALDR ----
        cmd_text = cmd_map[name]
        results = []
        errors = []

        for b in target_beams:
            key = f"baldr{b}"
            sock = self.servers.get(key)
            if sock is None:
                errors.append(f"{key}: not connected")
                continue

            try:
                sock.send_string(cmd_text)
                raw = sock.recv_string()  # commander replies as string (often JSON)
            except Exception as e:
                errors.append(f"{key}: ZMQ error: {e}")
                continue

            # Try to interpret the reply (JSON preferred; else plain text)
            ok = True
            details = None
            if not raw:
                ok = False
                details = "empty reply"
            else:
                try:
                    rep = json.loads(raw)
                    if isinstance(rep, dict):
                        # Common patterns: {"ok":true}, {"ok":false,"error":"..."},
                        # or {"reply":{"content":"OK"}} (or an error string)
                        if rep.get("ok") is False:
                            ok = False
                            details = rep.get("error") or rep
                        elif "reply" in rep and isinstance(rep["reply"], dict):
                            content = rep["reply"].get("content", "")
                            if isinstance(content, str) and content.upper().startswith(
                                "ERROR"
                            ):
                                ok = False
                                details = content
                            else:
                                details = content or "ok"
                        else:
                            # Unknown JSON shape; treat as OK and echo object
                            details = rep
                    else:
                        details = rep
                except json.JSONDecodeError:
                    # Non-JSON reply; consider it error only if it contains 'error'
                    if isinstance(raw, str) and "error" in raw.lower():
                        ok = False
                    details = raw  # send to Heimdallr the integration time.

            if ok:
                results.append(f"{key}: OK")
            else:
                errors.append(f"{key}: {details}")

        if errors:
            if results:
                return self.create_response(
                    f"ERROR: partial ({'; '.join(results)}; errors: {', '.join(errors)})"
                )
            return self.create_response(f"ERROR: {', '.join(errors)}")

        # All good
        return self.create_response("OK")

    def handle_hdlr_rts(self, command):
        """
        Any formatting needs to be done here

        simplest command: servo off
        """
        cmd_name = command.get("name", "").lower()

        if cmd_name == "hdlr_servo":
            parameters = command.get("parameters", [])
            servo_state = None
            for param in parameters:
                if param.get("name") == "state":
                    servo_state = param.get("value")
                    break
            if servo_state not in ["on", "off", "lacour"]:
                return self.create_response(
                    "ERROR: 'state' parameter must be 'on', 'off', or 'lacour'"
                )

            server = self.servers.get("hdlr")
            if server is None:
                return self.create_response("ERROR: hdlr server not connected")

            try:
                server.send_string(f"servo {servo_state}")
                reply = server.recv_string()
                if reply.upper().startswith("ERROR"):
                    return self.create_response(f"ERROR: hdlr response: {reply}")
            except Exception as e:
                return self.create_response(f"ERROR: ZMQ error: {e}")

            return self.create_response("OK")
        elif cmd_name == "hdlr_foreground":
            # Implement expstatus logic here
            self.servers["hdlr"].send_string(f'servo "off"')
            res = self.servers["hdlr"].recv_string()
            self.servers["hdlr"].send_string(f'offload "off"')
            res = self.servers["hdlr"].recv_string()

            self.servers["hdlr"].send_string(f"foreground")
            res = self.servers["hdlr"].recv_string()
            if res.upper().startswith("ERROR"):
                return self.create_response(f"ERROR: hdlr response: {res}")

            return self.create_response(res)
        elif cmd_name == "hdlr_fringe_search":
            # Execute a default fringe search, which is just 'offload "gd"'
            # plus other standard parameters.
            server = self.servers.get("hdlr")
            if server is None:
                return self.create_response("ERROR: hdlr server not connected")

            try:
                # None of these commmand have any responses.
                logging.info("Starting sending commands to hdlr server")
                server.send_string('servo "off"')
                server.recv_string()
                time.sleep(0.1)
                server.send_string("foreground 0")
                server.recv_string()
                time.sleep(0.1)
                server.send_string("offload_time 10")
                server.recv_string()
                time.sleep(0.1)
                server.send_string("set_gd_boxcar 64")
                server.recv_string()
                time.sleep(0.1)
                server.send_string(f"default_gains")
                server.recv_string()
                time.sleep(0.1)
                server.send_string('offload "gd"')
                server.recv_string()

                logging.info("Done")
            except Exception as e:
                return self.create_response(f"ERROR: ZMQ error: {e}")

            return self.create_response("OK")

    def abort(self):
        # abort the process that was run using subprocess by sending SIGINT
        # for all processes in self.scripts_running
        for pid, process in self.scripts_running.items():
            process.send_signal(signal.SIGINT)

            # Wait for the process to complete and get output
            stdout, stderr = process.communicate()
            logging.info("Signit process output:")
            logging.info(stdout)
            if stderr:
                logging.error("Signit process errors:")
                logging.error(stderr)

    def handle_script(self, command):
        """
        Handle script commands, e.g., s_h-autoalign
        """

        def _param_value(params_obj, key, default=None):
            # Accepts the RTS list format and (optionally) a dict format, looks for a paraticular parameter
            # e.g. to get the beam parameter >beam_raw = _param_value(params, "beam", 0)
            if isinstance(params_obj, list):
                for item in params_obj:
                    if isinstance(item, dict) and item.get("name") == key:
                        return item.get("value", default)
                return default
            if isinstance(params_obj, dict):
                return params_obj.get(key, default)
            return default

        def _log_subprocess_output(process, prefix=None):
            """Read and log output from process in real time."""

            def log_stream(stream, log_func, stream_name):
                for line in iter(stream.readline, b""):
                    try:
                        decoded = line.decode(errors="replace").rstrip()
                    except Exception:
                        decoded = str(line).rstrip()
                    if prefix:
                        log_func(f"[{prefix} {stream_name}] {decoded}")
                    else:
                        log_func(f"[{stream_name}] {decoded}")
                stream.close()

            threads = []
            if process.stdout:
                t_out = threading.Thread(
                    target=log_stream, args=(process.stdout, logging.info, "stdout")
                )
                t_out.daemon = True
                t_out.start()
                threads.append(t_out)
            if process.stderr:
                t_err = threading.Thread(
                    target=log_stream, args=(process.stderr, logging.error, "stderr")
                )
                t_err.daemon = True
                t_err.start()
                threads.append(t_err)
            return threads

        command_name = command.get("name", "").lower()
        # parameters = command.get("parameters", [])
        if command_name == "s_h-autoalign":
            process = subprocess.Popen(
                ["h-autoalign", "-a", "ia", "-o", "mcs", "-b", "K1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _log_subprocess_output(process, prefix=command_name)
            logging.info("Started s_h-autoalign script process.")
        elif command_name == "s_h-shutter":
            # command requires beam_time and dark_time parameters
            if _param_value(command.get("parameters", []), "beam-time") is None:
                return self.create_response("ERROR: beam_time parameter is required")
            if _param_value(command.get("parameters", []), "dark-time") is None:
                return self.create_response("ERROR: dark_time parameter is required")

            cmd = [
                "h-shutter",
                "--beam-time",
                str(_param_value(command.get("parameters", []), "beam-time")),
                "--dark-time",
                str(_param_value(command.get("parameters", []), "dark-time")),
            ]
            logging.info(cmd)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _log_subprocess_output(process, prefix=command_name)
            logging.info("Started s_h-shutter script process.")

        elif command_name == "s_b-autoalign":
            script = Path(
                "/home/asg/Progs/repos/dcs/dcs/cmd_scripts/b_autoalign_onsky.py"
            )

            for beam in [1, 2, 3, 4]:
                cmd = [
                    sys.executable,
                    str(script),
                    "--beam",
                    f"{beam}",
                    "--output",
                    "mcs",
                    "--mode",
                    "bright",
                    "--savepath",
                    "/home/asg/Pictures/baldr_pup_detect_onsky.png",
                ]
                process = subprocess.Popen(
                    cmd,
                    cwd=str(script.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                _log_subprocess_output(process, prefix=f"{command_name}-beam{beam}")
                logging.info(f"Started s_b-autoalign script for beam {beam}.")
                time.sleep(0.8)
        else:
            logging.error(f"Unknown script command '{command_name}'")
            return self.create_response(
                f"ERROR: Unknown script command '{command_name}'"
            )

        self.scripts_running[process.pid] = process
        return self.create_response("OK")

    def create_response(self, content):
        # Get the current UTC time
        current_utc_time = datetime.datetime.utcnow()
        # Format the UTC time
        return {
            "reply": {
                "time": current_utc_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "content": content,
            }
        }

    def setup(self, command):
        # Implement setup logic here
        parameters = command.get("parameters", [])
        # Validate and process parameters as needed
        dit = 0
        ndit = 0

        for param in parameters:
            name = param.get("name")
            value = param.get("value")
            logging.info(f"Setup parameter: {name} = {value}")
            # Add logic to handle each parameter as needed
            # DIT, NDIT, NWORESET, etc.
            if name == "DET.DIT":
                # Handle DIT parameter
                fps = 1 / value
                self.servers["cam_server"].send_string(f"set_fps {fps:.1f}")
                logging.info(self.servers["cam_server"].recv().decode("ascii"))
                # Also send this to heimdallr the integration time.
                self.servers["hdlr"].send_string(f"set_dit {value}")
                res = self.servers["hdlr"].recv_string()
                if res.upper().startswith("ERROR"):
                    return self.create_response(f"ERROR: hdlr response: {res}")
                dit = value
            elif name == "DET.NWORESET":
                try:
                    value = int(value)
                except ValueError:
                    return self.create_response(
                        f"ERROR: NWORESET value must be an integer"
                    )
                if value <= 2:
                    self.servers["cam_server"].send_string(
                        f'cli "set mode globalresetcds"'
                    )
                    logging.info(self.servers["cam_server"].recv().decode("ascii"))
                elif value < 500:
                    self.servers["cam_server"].send_string(
                        f'cli "set mode globalresetbursts"'
                    )
                    logging.info(self.servers["cam_server"].recv().decode("ascii"))
                    self.servers["cam_server"].send_string(
                        f'cli "set nbreadworeset {value}"'
                    )
                    logging.info(self.servers["cam_server"].recv().decode("ascii"))
                else:
                    return self.create_response(
                        f"ERROR: NWORESET value {value} is higher than the max (500)"
                    )
            elif name == "DET.GAIN":
                try:
                    value = int(value)
                except ValueError:
                    return self.create_response(f"ERROR: GAIN value must be an integer")
                if value < 0 or value > 100:
                    return self.create_response(
                        f"ERROR: GAIN value {value} is out of range (0-100)"
                    )
                self.servers["cam_server"].send_string(f"set_gain {value}")
                logging.info(self.servers["cam_server"].recv().decode("ascii"))
            elif name == "DET.NDIT":
                ndit = value
            elif name == "DET.EXP.NAME":
                pass
            else:
                # Handle other parameters as needed
                logging.warning(f"Unknown parameter: {name} = {value}")
                return self.create_response(f"ERROR: Unknown parameter '{name}'")
            logging.info("snoozing for a moment to try to fix gain reset issue")
            time.sleep(0.5)

        # If both NDIT and DIT are non-zero, we compute the total integration time.
        logging.info(f"Recv: DIT={dit}, NDIT={ndit} (total itime={dit*ndit})")
        if (dit != 0) and (ndit != 0):
            self.itime = dit * ndit

        return self.create_response("OK")

    def start(self, command):
        # check that the shutter script isn't running
        if "s_h-shutter" in self.scripts_running:
            return self.create_response("ERROR: s_h-shutter script is running")

        self.servers["cam_server"].send_string("save_mode 1")
        res = self.servers["cam_server"].recv_string()

        self.servers["hdlr"].send_string(f"set_itime {self.itime}")
        res = self.servers["hdlr"].recv_string()
        if res.upper().startswith("ERROR"):
            return self.create_response(f"ERROR: hdlr response: {res}")

        return self.create_response("OK")

    def expstatus(self, command):
        # Implement expstatus logic here

        # Send "expstatus" query to heimdallr, and return if the exposure is complete !!!
        # response is "integrating" or "success" or "failure" !!!
        self.servers["hdlr"].send_string(f"expstatus")
        res = self.servers["hdlr"].recv_string().replace('"', "")

        if res.upper().startswith("ERROR"):
            return self.create_response(f"ERROR: hdlr response: {res}")

        return self.create_response(res)

    # def foreground(self, command):


if __name__ == "__main__":
    server = BackEndServer()
    server.run()
