from typing import Any, Callable, Optional
from dataclasses import dataclass, field
import inspect

import numpy as np
import zmq
import time

import Cam
import DM
import AO
import LazyPirateZMQ


@dataclass
class Command:
    """Metadata for a command"""

    info: str
    func: Callable
    signature: inspect.Signature = field(init=False)
    parameters: list[inspect.Parameter] = field(init=False)

    def __post_init__(self):
        self.signature = inspect.signature(self.func)
        self.parameters = list(self.signature.parameters.values())


class BAOServer:
    def __init__(self, BAO, port):
        self.BAO = BAO

        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://192.168.10.2:{port}")

    def run(self):
        commands = {
            "servo": Command(
                info="Start the AO loop servo",
                func=self.BAO.servo,
            ),
            "take_dark": Command(
                info="Take a dark frame",
                func=self.BAO.take_dark,
            ),
            "take_interaction_matrix": Command(
                info="Acquire interaction matrix",
                func=self.BAO.take_interaction_matrix,
            ),
            "create_reconstructor": Command(
                info="Build linear reconstructor",
                func=self.BAO.create_reconstructor,
            ),
            "create_controller": Command(
                info="Create controller",
                func=self.BAO.create_controller,
            ),
            "take_ref": Command(
                info="Take reference image",
                func=self.BAO.take_ref,
            ),
        }

        while True:
            msg = self.sock.recv_string()
            parts = msg.strip().split()

            if not parts:
                self.sock.send_string("Empty command")
                self.BAO.run_iteration()
                continue

            cmd_name = parts[0]
            cmd = commands.get(cmd_name)
            if cmd is None:
                self.sock.send_string(
                    f"Unknown command '{cmd_name}'. Available: {', '.join(sorted(commands))}"
                )
                self.BAO.run_iteration()
                continue

            try:
                args, kwargs = self._parse_args(parts[1:])
                cast_args, cast_kwargs = self._coerce_types(cmd, args, kwargs)
                result = cmd.func(*cast_args, **cast_kwargs)
                self.sock.send_string(self._format_result(result))
            except Exception as exc:
                self.sock.send_string(f"Error: {exc}")

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
        if isinstance(result, np.ndarray):
            return f"array shape={result.shape} dtype={result.dtype}"
        return str(result)

