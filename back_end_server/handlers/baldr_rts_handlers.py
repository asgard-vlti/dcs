# handlers/baldr.py
from __future__ import annotations
import datetime
from typing import Any, Dict, List
import zmq

from rts_base import AbstractRTSTask, RTSState, RTSErr

# Baldr RTC ports per beam
BALDR_PORTS = {1: 6662, 2: 6663, 3: 6664, 4: 6665}

# ---------- helpers ----------
def _parse_beams(p: Dict[str, Any]) -> List[int]:
    """Accept 'beamid' or 'beam'; 0 => all beams; '1,3' => [1,3]."""
    val = p.get("beamid", p.get("beam"))
    if val is None:
        raise ValueError("missing 'beamid' (1-4) or 0 for all")
    if isinstance(val, str) and "," in val:
        beams = [int(s.strip()) for s in val.split(",") if s.strip()]
    else:
        beams = [int(val)]
    if beams == [0]:
        beams = [1, 2, 3, 4]
    for b in beams:
        if b not in BALDR_PORTS:
            raise ValueError(f"invalid beamid {b} (must be 1..4 or 0)")
    return beams

def _host(task: AbstractRTSTask) -> str:
    # Default host; override at server init via: server.rts_ctx.extras["baldr_host"] = "localhost"
    return (task.ctx.extras or {}).get("baldr_host", "mimir")

def _send(host: str, port: int, cmd: str, timeout_ms: int = 2000) -> str:
    """Send bare command name to the RTC. Swap to JSON if needed."""
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    try:
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, timeout_ms)
        s.setsockopt(zmq.SNDTIMEO, timeout_ms)
        s.connect(f"tcp://{host}:{port}")
        s.send_string(cmd)
        return s.recv_string()
    finally:
        s.close()

# ---------- generic handler ----------
class BaldrRTS(AbstractRTSTask):
    """
    Generic Baldr RTS handler.
    Uses incoming command["name"] (with optional 'bld_' stripped) as the string to send.
    Requires 'beamid' (or 'beam'); 0 = all beams.
    """

    def ok_to_run(self) -> bool:
        try:
            p = self.params_as_dict()
            beams = _parse_beams(p)
            self.metadata.update({
                "beams": beams,
                "targets": {b: BALDR_PORTS[b] for b in beams},
            })
            self.state, self.err = int(RTSState.READY), int(RTSErr.OK)
            return True
        except Exception as e:
            self.state, self.err = int(RTSState.FAILED), int(RTSErr.INVALID)
            self.metadata["error"] = str(e)
            return False

    def run(self, args=None) -> None:
        self.state = int(RTSState.RUNNING)
        raw = (self.command.get("name") or "").lower()
        cmd = raw[4:] if raw.startswith("bld_") else raw

        host = _host(self)
        results: Dict[int, Dict[str, Any]] = {}
        any_error = False

        for beam in self.metadata["beams"]:
            port = BALDR_PORTS[beam]
            try:
                reply = _send(host, port, cmd)
                results[beam] = {"ok": True, "reply": reply}
            except Exception as e:
                any_error = True
                results[beam] = {"ok": False, "error": str(e)}

        self.metadata["results"] = results
        self.metadata["finished_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

        if any_error:
            self.state = int(RTSState.FAILED)
            self.err = int(RTSErr.RUNTIME)
            self.metadata.setdefault("error", "one or more beams failed")
        else:
            self.state = int(RTSState.DONE)
            self.err = int(RTSErr.OK)

    def update_DB(self) -> None:
        # TODO: implement notifier when MCS/WAG integration is ready.
        # e.g., if callable(self.ctx.mcs_notify): self.ctx.mcs_notify(payload)
        return

    def abort(self) -> bool:
        # Default: mark aborted. Override in a subclass if a command needs special cleanup.
        self.state, self.err = int(RTSState.ABORTED), int(RTSErr.ABORTED)
        self.metadata["error"] = "aborted"
        return True

# ---------- optional per-command overrides (kept as examples; also TODO) ----------
class SaveTelemetryRTS(BaldrRTS):
    def update_DB(self) -> None:
        # TODO: push a lighter summary for telemetry saves when notifier exists.
        return

class OpenBaldrHORTS(BaldrRTS):
    def abort(self) -> bool:
        # TODO: add HO-specific cleanup if required.
        self.state, self.err = int(RTSState.ABORTED), int(RTSErr.ABORTED)
        self.metadata["error"] = "aborted (HO-specific)"
        return True

# ---------- registry hook ----------
def register(server) -> None:
    # Default all names to the generic handler
    default_names = [
        "open_baldr_lo", "close_baldr_lo",
        "open_baldr_ho", "close_baldr_ho",
        "open_all", "close_all",
        "save_telemetry",
    ]
    overrides = {
        "save_telemetry": SaveTelemetryRTS,  # uses TODO override above
        "open_baldr_ho": OpenBaldrHORTS,     # uses TODO override above
    }
    for name in default_names:
        cls = overrides.get(name, BaldrRTS)
        server.register_rts(name, cls)
