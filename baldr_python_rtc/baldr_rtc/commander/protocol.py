from __future__ import annotations

import json
from typing import Any, List, Tuple, Union

Json = Union[None, bool, int, float, str, List["Json"], dict]


def parse_message_to_command_and_args(message: str) -> Tuple[str, List[Json]]:
    message = message.strip()
    if not message:
        raise ValueError("Empty command.")

    if " " not in message:
        return message, []

    name, rest = message.split(" ", 1)
    rest = rest.strip()
    if rest == "":
        return name, []

    # JSON single token or list/object
    if rest[:1] in ("[", "{") or rest in ("null", "true", "false") or rest[:1] in ('"', "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        try:
            val = json.loads(rest)
            if isinstance(val, list):
                return name, val
            return name, [val]
        except Exception:
            pass

    wrapped = f"[{rest}]"
    arr = json.loads(wrapped)
    if not isinstance(arr, list):
        raise ValueError("Arguments must parse to a JSON array.")
    return name, arr
