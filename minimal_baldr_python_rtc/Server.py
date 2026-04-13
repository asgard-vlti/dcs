from typing import Any, Callable, Optional
from dataclasses import dataclass, field
import inspect
import logging
import pathlib
from datetime import datetime

import numpy as np
import zmq
import time

import Cam
import DM
import AO
import LazyPirateZMQ
from BaldrAO import BaldrAO
import json
import argparse

import consts

logger = logging.getLogger(__name__)


def setup_logging_for_beam(beam: int) -> pathlib.Path:
    log_dir = pathlib.Path(f"~/logs/minimal_baldr/beam_{beam}").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"BAO_log_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    return log_path


@dataclass
class Command:
    """Metadata for a command"""

    info: str
    func: Callable
    is_short: bool  # if long, the response should be "ok" immediately
    signature: inspect.Signature = field(init=False)
    parameters: list[inspect.Parameter] = field(init=False)

    def __post_init__(self):
        self.signature = inspect.signature(self.func)
        self.parameters = list(self.signature.parameters.values())


class BAOServer:
    def __init__(self, BAO: BaldrAO, port):
        self.BAO = BAO

        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://mimir:{port}")

        self.commands = {
            "servo": Command(
                info="Start the AO loop servo",
                func=self.BAO.servo,
                is_short=True,
            ),
            "take_dark": Command(
                info="Take a dark frame",
                func=self.BAO.take_dark,
                is_short=False,
            ),
            "take_interaction_matrix": Command(
                info="Acquire interaction matrix",
                func=self.BAO.take_interaction_matrix,
                is_short=False,
            ),
            "create_reconstructor": Command(
                info="Build linear reconstructor",
                func=self.BAO.create_reconstructor,
                is_short=False,
            ),
            "create_controller": Command(
                info="Create controller",
                func=self.BAO.create_controller,
                is_short=True,
            ),
            "take_ref": Command(
                info="Take reference image",
                func=self.BAO.take_ref,
                is_short=False,
            ),
            "save_state": Command(
                info="Save current state to a timestamped pickle file",
                func=self.BAO.save_state,
                is_short=True,
            ),
            "load_state": Command(
                info="Load state from a pickle file",
                func=self.BAO.load_state,
                is_short=True,
            ),
            "status": Command(
                info="Get current status of the system",
                func=self.BAO.get_status,
                is_short=True,
            ),
            "flatten_dm": Command(
                info="Flatten the DM (zero all actuators)",
                func=self.BAO.dm.flatten,
                is_short=True,
            ),
            "set_ki_gains": Command(
                info="Set integrator gains (ki) for all modes. ",
                func=self.BAO.set_ki_gains,
                is_short=True,
            ),
            "set_leaks": Command(
                info="Set leak factors for all modes. ",
                func=self.BAO.set_leaks,
                is_short=True,
            ),
            "set_Lmax": Command(
                info="Set the L_max parameter for the DM Laplacian limiter. ",
                func=self.BAO.set_L_max,
                is_short=True,
            ),
            "update_estimator_mask": Command(
                info="Update the mask used by the Strehl estimator based on current pupil image",
                func=self.BAO.update_estimator_mask,
                is_short=False,
            ),
            "set_open_threshold": Command(
                info="Set the open threshold for the Strehl estimator",
                func=self.BAO.set_open_threshold,
                is_short=True,
            ),
            "set_close_threshold": Command(
                info="Set the close threshold for the Strehl estimator",
                func=self.BAO.set_close_threshold,
                is_short=True,
            ),
            "get_controller_params": Command(
                info="Get current integrator gains (ki) and leakage for all modes. ",
                func=self.BAO.get_controller_params,
                is_short=True,
            ),
            "command_names": Command(
                info="command_names - list all available commands",
                func=lambda: json.dumps(
                    [cmd_name for cmd_name in self.commands.keys()]
                ),
                is_short=True,
            ),
        }

    def run(self):
        while True:
            try:
                msg = self.sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                self.BAO.run_iteration()
                continue

            parts = msg.strip().split()

            if not parts:
                self.sock.send_string("Empty command")
            else:
                cmd_name = parts[0]
                cmd = self.commands.get(cmd_name)
                if cmd is None:
                    self.sock.send_string(
                        f"Unknown command '{cmd_name}'. Available: {', '.join(sorted(self.commands))}"
                    )
                else:
                    try:
                        if not cmd.is_short:
                            self.sock.send_string("OK")
                        args, kwargs = self._parse_args(parts[1:])
                        cast_args, cast_kwargs = self._coerce_types(cmd, args, kwargs)
                        result = cmd.func(*cast_args, **cast_kwargs)
                        if cmd.is_short:
                            self.sock.send_string(self._format_result(result))
                    except Exception as exc:
                        self.sock.send_string(f"Error: {exc}")
                        logger.exception("Error while executing command '%s'", cmd_name)

            self.BAO.run_iteration()

    @staticmethod
    def _parse_args(tokens: list[str]) -> tuple[list[str], dict[str, str]]:
        args: list[str] = []
        kwargs: dict[str, str] = {}
        for token in tokens:
            if "=" in token:
                key, value = token.split("=", 1)
                kwargs[key] = value
            else:
                args.append(token)
        return args, kwargs

    def _coerce_types(
        self, cmd: Command, args: list[str], kwargs: dict[str, str]
    ) -> tuple[list[Any], dict[str, Any]]:
        if len(args) > len(cmd.parameters):
            raise ValueError("Too many positional arguments")

        cast_args: list[Any] = []
        cast_kwargs: dict[str, Any] = {}

        for idx, raw_value in enumerate(args):
            param = cmd.parameters[idx]
            cast_args.append(self._cast_value(raw_value, param))

        for key, raw_value in kwargs.items():
            if key not in cmd.signature.parameters:
                raise ValueError(f"Unknown argument '{key}'")
            param = cmd.signature.parameters[key]
            cast_kwargs[key] = self._cast_value(raw_value, param)

        # validate required arguments and reject duplicate position+keyword
        seen_positionals = {param.name for param in cmd.parameters[: len(cast_args)]}
        duplicate_keys = seen_positionals.intersection(cast_kwargs.keys())
        if duplicate_keys:
            dup = ", ".join(sorted(duplicate_keys))
            raise ValueError(f"Argument(s) provided twice: {dup}")

        try:
            cmd.signature.bind(*cast_args, **cast_kwargs)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc

        return cast_args, cast_kwargs

    @staticmethod
    def _cast_value(value: str, param: inspect.Parameter):
        annotation = param.annotation
        empty = inspect.Signature.empty
        if annotation is empty:
            if param.default is not empty:
                annotation = type(param.default)
            else:
                return value

        if annotation is bool:
            lowered = value.lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            raise ValueError(
                f"Invalid boolean for '{param.name}': '{value}' "
                "(expected true/false, 1/0, yes/no, on/off)"
            )

        try:
            return annotation(value)
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _format_result(result: Any) -> str:
        if result is None:
            return "OK"
        if isinstance(result, dict):
            return json.dumps(result)
        if isinstance(result, np.ndarray):
            return f"array shape={result.shape} dtype={result.dtype}"
        return str(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAO ZMQ Server")
    parser.add_argument(
        "--beam", type=int, required=True, help="Beam number to control"
    )
    args = parser.parse_args()

    log_path = setup_logging_for_beam(args.beam)
    logger.info("Logging to %s", log_path)

    bao = BaldrAO(args.beam)
    server = BAOServer(bao, port=consts.BEAM_TO_PORT[args.beam])
    logger.info(
        "Starting BAO server for beam %s on port %s",
        args.beam,
        consts.BEAM_TO_PORT[args.beam],
    )
    server.run()
