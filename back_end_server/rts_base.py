# rts_base.py
from __future__ import annotations
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set

import subprocess


# ---- integer states / errors (keep attributes as plain ints) ----
class RTSState(IntEnum):
    INIT = 0
    READY = 1
    RUNNING = 2
    DONE = 3
    ABORTED = 4
    FAILED = 5


class RTSErr(IntEnum):
    OK = 0
    INVALID = 1
    RUNTIME = 2
    ABORTED = 3
    UNKNOWN = 99


# ---- optional context you can pass from the server ----
@dataclass
class RTSContext:
    """
    Carry optional services/config for tasks.
    - mcs_notify: callable to notify WAG/MCS (payload: dict) -> dict ack
    - allowed_commands: optional allowlist of RTS names
    - extras: stash anything deployment-specific (e.g., endpoints)
    """

    mcs_notify: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    allowed_commands: Set[str] = field(default_factory=set)
    extras: Dict[str, Any] = field(default_factory=dict)


class AbstractRTSTask(ABC):
    """
    Abstract base for RTS-type commands (fire-and-forget friendly).

    Server-side scheduling:
      - Server calls ok_to_run() synchronously.
      - If OK, server enqueues task into a bounded thread pool.
      - Worker thread then calls run() and update_DB().
      - abort() can be called asynchronously by the server if you expose that API.

    Methods to implement:
        ok_to_run()  -> bool
        run(args)    -> None
        update_DB()  -> None
        abort()      -> bool

    Public attributes (ints & dict by contract):
        state: int
        err: int
        metadata: dict
    """

    # public attributes (requested shapes)
    state: int
    err: int
    metadata: Dict[str, Any]

    def __init__(
        self, command: Dict[str, Any], ctx: Optional[RTSContext] = None
    ) -> None:
        self.command: Dict[str, Any] = command or {}
        self.ctx: RTSContext = ctx or RTSContext()
        self.state = int(RTSState.INIT)
        self.err = int(RTSErr.OK)
        self.metadata = {
            "command_name": self.command.get("name"),
            "command_time": self.command.get("time"),
            "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # ---- abstract lifecycle (no replies returned here) ----
    @abstractmethod
    def ok_to_run(self) -> bool:
        """Validate inputs/resources; set state/err/metadata; return True if ready."""
        raise NotImplementedError

    @abstractmethod
    def run(self, args: Optional[dict | list] = None) -> None:
        """Do the work (blocking in worker thread). Update state/err/metadata."""
        raise NotImplementedError

    @abstractmethod
    def update_DB(self) -> None:
        """Notify/commit result to WAG/MCS. Use ctx.mcs_notify if provided."""
        raise NotImplementedError

    @abstractmethod
    def abort(self) -> bool:
        """Best-effort abort. Return True if an abort was initiated/handled."""
        raise NotImplementedError

    # ---- tiny helpers (optional, handy in subclasses) ----
    def make_reply(self, content: str) -> Dict[str, Any]:
        """Build the standard wire reply (server can also use its own helper)."""
        now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        return {"reply": {"time": now, "content": content}}

    def mcs_notify(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call the MCS notify function if provided, else return a dummy ack."""
        if self.ctx.mcs_notify:
            return self.ctx.mcs_notify(payload)
        return {"ok": True, "note": "no mcs_notify function provided"}

    def params_as_dict(self) -> Dict[str, Any]:
        """Normalize WAG parameters [{'name','value'}, ...] -> {name: value}."""
        out: Dict[str, Any] = {}
        for p in self.command.get("parameters", []) or []:
            n = (p.get("name") or "").strip()
            if n:
                out[n] = p.get("value")
        return out

    def set_error(self, msg: str, err: RTSErr = RTSErr.INVALID) -> None:
        self.state = int(RTSState.FAILED)
        self.err = int(err)
        self.metadata["error"] = msg

    def set_done(self) -> None:
        self.state = int(RTSState.DONE)
        self.err = int(RTSErr.OK)


# """

# Abstract class that recieves a command from WAG
# and does somethings and then updates (or tells wag to update via MCS) database

# Methods
#     ok_to_run() -> bool
#     run(args) -> dict
#     update_DB() -> dict
#     abort() -> bool

# Attributes
#     state : int
#     err : int
#     metadata : dict


# follows Asgard Top-Level Control Software User and Maintenance Manual
# particularly section "8.5.6. RTS command"
# """
# # rts_base.py
# from __future__ import annotations
# from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
# from enum import IntEnum
# from typing import Any, Dict, Optional, Callable, Set

# def create_response(self, content):
#     # Get the current UTC time
#     current_utc_time = datetime.datetime.utcnow()
#     # Format the UTC time
#     return {
#         "reply": {
#             "time": current_utc_time.strftime("%Y-%m-%dT%H:%M:%S"),
#             "content": content,
#         }
#     }

# # Int-valued states/errors (attributes on tasks remain plain ints)
# class RTSState(IntEnum):
#     INIT    = 0
#     READY   = 1
#     RUNNING = 2
#     DONE    = 3
#     ABORTED = 4
#     FAILED  = 5


# class RTSErr(IntEnum):
#     OK       = 0
#     INVALID  = 1
#     RUNTIME  = 2
#     ABORTED  = 3
#     UNKNOWN  = 99


# @dataclass
# class RTSContext:
#     """Lightweight context you can pass into tasks (optional)."""
#     mcs_notify: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
#     allowed_commands: Set[str] = field(default_factory=set)


# class AbstractRTSTask(ABC):
#     """
#     Abstract base for RTS-type commands.

#     Methods to override in subclasses:
#         ok_to_run()  -> bool
#         run(args)    -> dict
#         update_DB() -> dict
#         abort()      -> bool

#     Attributes :
#         state: int
#         err: int
#         metadata: dict
#         sockets : dict
#     """

#     # public attributes
#     state: int
#     err: int
#     metadata: Dict[str, Any]
#     sockets : Dict[str, Any]

#     def __init__(self, command: Dict[str, Any], ctx: Optional[RTSContext] = None) -> None:
#         self.command: Dict[str, Any] = command or {}
#         self.ctx: RTSContext = ctx or RTSContext()
#         self.state = int(RTSState.INIT)
#         self.err = int(RTSErr.OK)
#         self.metadata = {}  # subclasses can populate as they run
#         self.sockets = {
#             "cam_server" :    6667,
#             "DM_server"   :    6666,
#             "hdlr"         :   6660,
#             "hdlr_align"   :   6661,
#             "baldr1"        :   6662,
#             "baldr2"        :   6663,
#             "baldr3"        :   6664,
#             "baldr4"        :   6665,
#             "MCS"           :   7020,
#             "ICS"           :   5555,
#             "RTD"          :  7000,
#         }
#     # ------- abstract lifecycle to be implemented by concrete commands -------
#     @abstractmethod
#     def ok_to_run(self) -> bool:
#         """Validate inputs/resources; set self.state/err; return True if ready."""
#         raise NotImplementedError

#     @abstractmethod
#     def run(self, args: Optional[dict | list] = None) -> dict:
#         """Execute the command (blocking or orchestrating); update state/err/metadata; return metadata."""
#         return create_response(self, content="OK") # must return this format of respones (the "conent" can be different depending on particular run case)

#     @abstractmethod
#     def update_DB(self) -> dict:
#         """Notify/commit state to WAG/MCS (shape up to your system); return an ack dict."""
#         raise NotImplementedError

#     @abstractmethod
#     def abort(self) -> bool:
#         """Best-effort abort; return True if an abort was initiated/handled."""
#         raise NotImplementedError

#     # =============================================================
#     # This is the main thing that gets called for every command call
#     def run_command(self,cmd) -> dict:

#         if self.ok_to_run():
#             return self.run( self.command.get("parameters",{})) # update_DB should be applied within run() for particular class instance
#             # run must return
#         else:
#             self.state = int(RTSState.FAILED) # update the internal state to FAILED
#             return create_response(self, content="ERROR") # return an error response
#     # =============================================================


# # example
# class OpenAO(AbstractRTSTask):
#     def ok_to_run(self) -> bool:
#         # validate command/params here
#         self.state = int(RTSState.READY)
#         self.err = int(RTSErr.OK)
#         return True

#     def run(self, args=None) -> dict:
#         self.state = int(RTSState.RUNNING)


#         # do work...
#         self.state = int(RTSState.DONE)
#         self.err = int(RTSErr.OK)
#         return self.metadata

#     def update_DB(self) -> dict:
#         # call MCS here if desired
#         return {"ok": True}

#     def abort(self) -> bool:
#         self.state = int(RTSState.ABORTED)
#         self.err = int(RTSErr.ABORTED)
#         return True


class S_HDLR_AUTO_ALIGN(AbstractRTSTask):
    def ok_to_run(self) -> bool:
        # validate command/params here
        self.state = int(RTSState.READY)
        self.err = int(RTSErr.OK)
        return True

    def run(self, args=None) -> dict:
        self.state = int(RTSState.RUNNING)

        subprocess.Popen(["h-autoalign", "-a", "ia", "-o", "mcs"])

        self.state = int(RTSState.DONE)
        self.err = int(RTSErr.OK)
        return self.metadata

    def update_DB(self) -> dict:
        self.mcs_notify(
            {
                "command": {
                    "name": "write",
                    "time": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                    "parameter": [
                        {
                            "name": "hdlr_complete",
                            "value": 0,
                            "requester": "mimir",
                        },
                    ],
                }
            }
        )

        return {"ok": True}

    def abort(self) -> bool:
        self.state = int(RTSState.ABORTED)
        self.err = int(RTSErr.ABORTED)
        return True
