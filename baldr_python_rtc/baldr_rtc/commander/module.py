from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

Json = Union[None, bool, int, float, str, List["Json"], dict]


@dataclass
class ArgumentSpec:
    name: str
    type: str
    default_value: Optional[Json] = None


@dataclass
class CommandSpec:
    name: str
    description: str = ""
    arguments: List[ArgumentSpec] = field(default_factory=list)
    return_type: Json = "any"


class Module:
    def __init__(self) -> None:
        self._specs: Dict[str, CommandSpec] = {}
        self._fns: Dict[str, Callable[[List[Json]], Json]] = {}

    def def_command(
        self,
        name: str,
        fn: Callable[[List[Json]], Json],
        *,
        description: str = "",
        arguments: Optional[List[ArgumentSpec]] = None,
        return_type: Json = "any",
    ) -> None:
        self._specs[name] = CommandSpec(
            name=name,
            description=description or "",
            arguments=arguments or [],
            return_type=return_type,
        )
        self._fns[name] = fn

    def execute(self, name: str, args: Optional[List[Json]] = None) -> Json:
        if name not in self._fns:
            return {"error": f"Command '{name}' not found."}
        if args is None:
            args = []
        if not isinstance(args, list):
            return {"error": "Arguments must be a JSON array."}
        try:
            return self._fns[name](args)
        except Exception as e:
            return {"error": str(e)}
