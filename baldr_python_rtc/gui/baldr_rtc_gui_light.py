#!/usr/bin/env python3
"""Lightweight Baldr RTC control GUI (no telemetry polling, no plots).

Per-beam controls:
- OPEN_LO   -> open_baldr_LO
- OPEN_HO   -> open_baldr_HO
- CLOSE_LO  -> close_baldr_LO
- CLOSE_HO  -> close_baldr_HO
- PHOT      -> update_N0_runtime

Gain controls (assumes leaky integrator everywhere):
- ki_LO, rho_LO, ki_HO, rho_HO
  Each has: indices ("all" or e.g. "0,1,4-7") + value + Update button.
  Sends: set_<lo|ho>_gain [<param>, <indices|all>, <value>]

Dependencies: PyQt5, pyzmq
"""

from __future__ import annotations

import sys
import json
import argparse
from dataclasses import dataclass

import zmq

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFrame
)


# ---------------- ZMQ commander client ----------------
class CommanderClient:
    def __init__(self, endpoint: str, *, rcv_timeout_ms: int = 800, snd_timeout_ms: int = 800):
        self.ctx = zmq.Context.instance()
        self.endpoint = endpoint
        self.rcv_timeout_ms = int(rcv_timeout_ms)
        self.snd_timeout_ms = int(snd_timeout_ms)
        self._make_sock()

    def _make_sock(self) -> None:
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.rcv_timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.snd_timeout_ms)
        self.sock.connect(self.endpoint)

    def _reset(self) -> None:
        try:
            self.sock.close(0)
        except Exception:
            pass
        self._make_sock()

    def send_raw(self, msg: str) -> str:
        try:
            self.sock.send_string(msg)
            return self.sock.recv_string()
        except zmq.error.Again:
            self._reset()
            return ""
        except zmq.ZMQError:
            self._reset()
            return ""


# ------------- helpers -------------

def _parse_indices(s: str) -> str:
    """Validate and normalize an indices spec, returning a STRING.

    Accepts:
      - "all" (case-insensitive)
      - comma-separated ints: "0, 1, 5"
      - ranges: "0-7" or mixed "0-3,8,10-12"

    Returns either "all" or a normalized string like "1,2,3,4" or "0-3,8,10-12".
    Raises ValueError on invalid input.
    """
    t = (s or "").strip()
    if not t or t.lower() == "all":
        return "all"

    # normalize: remove whitespace around commas/dashes
    # and validate tokens are ints or int-int ranges.
    parts = [p.strip() for p in t.split(",") if p.strip()]
    norm_parts: list[str] = []
    for p in parts:
        if "-" in p:
            a, b, *rest = [x.strip() for x in p.split("-")]
            if rest or a == "" or b == "":
                raise ValueError(f"Bad range token: {p!r}")
            int(a); int(b)  # validate
            norm_parts.append(f"{int(a)}-{int(b)}")
        else:
            norm_parts.append(str(int(p)))
    return ",".join(norm_parts)

@dataclass(frozen=True)
class GainSpec:
    label: str          # e.g. "ki_LO"
    band: str           # "lo" or "ho"
    param: str          # "ki" or "rho"


# ---------------- per-beam control panel ----------------
class BeamControlPanel(QWidget):
    def __init__(self, *, beam: int, endpoint: str):
        super().__init__()
        self.beam = int(beam)
        self.endpoint = endpoint
        self.client = CommanderClient(endpoint)

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout()

        title = QLabel(f"Beam {self.beam}  —  {self.endpoint}")
        title.setStyleSheet("font-weight: 600;")
        root.addWidget(title)

        # ---- quick actions ----
        quick = QGridLayout()

        self.open_lo_btn = QPushButton("OPEN_LO")
        self.open_lo_btn.clicked.connect(lambda: self._send_simple("open_baldr_LO"))

        self.open_ho_btn = QPushButton("OPEN_HO")
        self.open_ho_btn.clicked.connect(lambda: self._send_simple("open_baldr_HO"))

        self.close_lo_btn = QPushButton("CLOSE_LO")
        self.close_lo_btn.clicked.connect(lambda: self._send_simple("close_baldr_LO"))

        self.close_ho_btn = QPushButton("CLOSE_HO")
        self.close_ho_btn.clicked.connect(lambda: self._send_simple("close_baldr_HO"))

        self.phot_btn = QPushButton("UPDATE PHOTOMETRY (ON CLEAR PUPIL)")
        self.phot_btn.clicked.connect(lambda: self._send_simple("update_N0_runtime"))
        # I0 update 
        self.i0_btn = QPushButton("UPDATE ZWFS INTENS. SETPOINT")
        self.i0_btn.clicked.connect(lambda: self._send_simple("update_I0_runtime"))

        quick.addWidget(self.open_lo_btn, 0, 0)
        quick.addWidget(self.open_ho_btn, 0, 1)
        quick.addWidget(self.close_lo_btn, 1, 0)
        quick.addWidget(self.close_ho_btn, 1, 1)
        #quick.addWidget(self.phot_btn, 2, 0, 1, 2)
        quick.addWidget(self.phot_btn, 2, 0)
        quick.addWidget(self.i0_btn, 2, 1)
        root.addLayout(quick)

        # separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        # ---- gain controls ----
        root.addWidget(QLabel("Leaky integrator gains"))

        grid = QGridLayout()
        grid.addWidget(QLabel("Gain"), 0, 0)
        grid.addWidget(QLabel("Indices (all | 0,1,4-7)"), 0, 1)
        grid.addWidget(QLabel("Value"), 0, 2)
        grid.addWidget(QLabel(""), 0, 3)

        self._gain_rows: list[tuple[GainSpec, QLineEdit, QLineEdit, QPushButton]] = []

        specs = [
            GainSpec("ki_LO", band="lo", param="ki"),
            GainSpec("rho_LO", band="lo", param="rho"),
            GainSpec("ki_HO", band="ho", param="ki"),
            GainSpec("rho_HO", band="ho", param="rho"),
        ]

        for r, spec in enumerate(specs, start=1):
            lab = QLabel(spec.label)
            idx = QLineEdit("all")
            val = QLineEdit("")
            btn = QPushButton("Update")
            btn.clicked.connect(lambda _=False, s=spec, i=idx, v=val: self._send_gain(s, i.text(), v.text()))

            grid.addWidget(lab, r, 0)
            grid.addWidget(idx, r, 1)
            grid.addWidget(val, r, 2)
            grid.addWidget(btn, r, 3)

            self._gain_rows.append((spec, idx, val, btn))

        root.addLayout(grid)

        # separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep2)

        # ---- raw command line ----
        root.addWidget(QLabel("Commander"))

        cmdrow = QHBoxLayout()
        self.cmd_in = QLineEdit()
        self.cmd_in.setPlaceholderText('e.g. status   |   set_lo_gain ["ki","all",0.05]')
        self.cmd_send = QPushButton("Send")
        self.cmd_send.clicked.connect(self._send_cmd)
        cmdrow.addWidget(self.cmd_in)
        cmdrow.addWidget(self.cmd_send)
        root.addLayout(cmdrow)

        self.cmd_out = QTextEdit()
        self.cmd_out.setReadOnly(True)
        self.cmd_out.setMinimumHeight(180)
        root.addWidget(self.cmd_out)

        self.setLayout(root)

    def _log(self, s: str) -> None:
        self.cmd_out.append(s)

    def _send_simple(self, cmd: str) -> None:
        raw = self.client.send_raw(cmd)
        if raw:
            self._log(f"> {cmd}\n{raw}\n")
        else:
            self._log(f"> {cmd}\n(timeout)\n")

    def _send_gain(self, spec: GainSpec, indices_text: str, value_text: str) -> None:
        try:
            idx = _parse_indices(indices_text)
        except Exception as e:
            self._log(f"[gain] {spec.label}: bad indices {indices_text!r} ({e})")
            return

        try:
            val = float(value_text)
        except Exception:
            self._log(f"[gain] {spec.label}: bad value {value_text!r} (expected float)")
            return

        cmd = f"set_{spec.band}_gain"
        payload = [spec.param, idx, val]
        msg = f"{cmd} {json.dumps(payload)}"

        raw = self.client.send_raw(msg)
        if raw:
            self._log(f"> {msg}\n{raw}\n")
        else:
            self._log(f"> {msg}\n(timeout)\n")

    def _send_cmd(self) -> None:
        msg = self.cmd_in.text().strip()
        if not msg:
            return
        reply = self.client.send_raw(msg)
        self._log(f"> {msg}\n{reply}\n")
        self.cmd_in.clear()


# ---------------- main window ----------------
class MainWindow(QMainWindow):
    def __init__(self, panels: list[BeamControlPanel]):
        super().__init__()
        self.setWindowTitle("Baldr RTC Control GUI (light)")
        w = QWidget()
        layout = QHBoxLayout()
        for p in panels:
            layout.addWidget(p)
        w.setLayout(layout)
        self.setCentralWidget(w)


from baldr_python_rtc.scripts.baldr_server import beam2port

def default_socket_for_beam(beam: int) -> str:
    return f"tcp://127.0.0.1:{beam2port[int(beam)]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", type=str, default="1", help="comma-separated beams, e.g. 1 or 1,2,3,4")
    ap.add_argument("--socket", type=str, default="", help="override single socket (only if one beam)")
    ap.add_argument("--rcv-timeout-ms", type=int, default=800)
    ap.add_argument("--snd-timeout-ms", type=int, default=800)
    args = ap.parse_args()

    beams = [int(x) for x in args.beams.split(",") if x.strip()]
    if args.socket and len(beams) != 1:
        raise SystemExit("--socket only allowed with a single beam")

    app = QApplication(sys.argv)

    panels: list[BeamControlPanel] = []
    for b in beams:
        endpoint = args.socket if args.socket else default_socket_for_beam(b)
        p = BeamControlPanel(beam=b, endpoint=endpoint)
        # patch timeouts if user changed via CLI
        p.client.rcv_timeout_ms = int(args.rcv_timeout_ms)
        p.client.snd_timeout_ms = int(args.snd_timeout_ms)
        p.client._reset()
        panels.append(p)

    win = MainWindow(panels)
    win.resize(1200, 520)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()