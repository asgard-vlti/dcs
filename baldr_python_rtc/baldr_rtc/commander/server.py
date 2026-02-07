from __future__ import annotations

import json
import threading
import zmq

from baldr_python_rtc.baldr_rtc.commander.protocol import parse_message_to_command_and_args
from baldr_python_rtc.baldr_rtc.commander.module import Module



class CommanderServer(threading.Thread):
    def __init__(self, endpoint: str, module: Module, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.endpoint = endpoint
        self.module = module
        self.stop_event = stop_event
        self._ctx = zmq.Context.instance()

    def run(self) -> None:
        sock = self._ctx.socket(zmq.REP)
        sock.bind(self.endpoint)

        while not self.stop_event.is_set():
            if sock.poll(timeout=100, flags=zmq.POLLIN) == 0:
                continue

            msg = sock.recv_string().strip()

            if 'poll_telem' not in msg:
                print(f"received: {msg}")

            if msg == "exit":
                sock.send_string("Exiting!")
                self.stop_event.set()
                break

            try:
                name, args = parse_message_to_command_and_args(msg)
                out = self.module.execute(name, args)
                sock.send_string(json.dumps(out))
            except Exception as e:
                sock.send_string(json.dumps({"error": str(e)}))

        sock.close(0)
