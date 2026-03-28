from __future__ import annotations

from typing import Any, Dict


def make_cmd(name: str, **payload: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": name}
    d.update(payload)
    return d
