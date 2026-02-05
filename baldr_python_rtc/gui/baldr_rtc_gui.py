# baldr_telem_gui.py
#
# Minimal Baldr RTC telemetry GUI:
# - polls commander "poll_telem" (JSON request)
# - can plot scalar time series or render latest 2D image / DM map
# - includes a command line to send raw commander commands
#
# deps: PyQt5, pyqtgraph, pyzmq, numpy

import sys, time, json, argparse
from collections import deque

import numpy as np
import zmq

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QSpinBox, QCheckBox, QLineEdit, QPushButton, QTextEdit
)
from PyQt5.QtCore import QTimer

import pyqtgraph as pg


# ---------------- ZMQ commander client ----------------
class CommanderClient:
    def __init__(self, endpoint: str, *, rcv_timeout_ms=500, snd_timeout_ms=500):
        self.ctx = zmq.Context.instance()
        self.endpoint = endpoint
        self.rcv_timeout_ms = int(rcv_timeout_ms)
        self.snd_timeout_ms = int(snd_timeout_ms)
        self._make_sock()

    def _make_sock(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.rcv_timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.snd_timeout_ms)
        self.sock.connect(self.endpoint)

    def _reset(self):
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

    def call(self, cmd: str, arg: str | None = None) -> dict:
        msg = cmd if arg is None else f"{cmd} {arg}"
        raw = self.send_raw(msg)
        if not raw:
            return {"ok": False, "error": "timeout"}
        try:
            return json.loads(raw)
        except Exception:
            return {"ok": False, "error": "non-json", "raw": raw}


# ------------- helpers -------------
def dm140_to_12x12(cmd140: np.ndarray, Nx_act=12) -> np.ndarray:
    # Insert NaNs at the 4 missing corners (matches your earlier helper style)
    v = np.asarray(cmd140, dtype=float).ravel().tolist()
    # indices in a 12x12 flattened array where corners live
    # (note: insert shifts subsequent indices; do descending order)
    corner_insert = [0, Nx_act - 1, Nx_act * (Nx_act - 1), Nx_act * Nx_act]
    out = list(v)
    for i in sorted(corner_insert, reverse=True):
        out.insert(i, np.nan)
    return np.array(out, dtype=float).reshape(Nx_act, Nx_act)


def try_square_reshape(vec: np.ndarray) -> np.ndarray | None:
    v = np.asarray(vec).ravel()
    n = v.size
    s = int(round(np.sqrt(n)))
    if s * s == n:
        return v.reshape(s, s)
    return None


def reduce_vector(v: np.ndarray, reducer: str, idx: int) -> float:
    x = np.asarray(v, dtype=float).ravel()
    if x.size == 0:
        return float("nan")
    r = reducer.lower().strip()
    if r == "mean":
        return float(np.nanmean(x))
    if r == "rms":
        return float(np.sqrt(np.nanmean(x * x)))
    if r == "index":
        j = int(np.clip(idx, 0, x.size - 1))
        return float(x[j])
    return float(np.nanmean(x))


# ---------------- per-beam plot panel ----------------
class BeamPanel(QWidget):
    def __init__(self, *, beam: int, endpoint: str, history: int = 1000, poll_hz: float = 10.0):
        super().__init__()
        self.beam = int(beam)
        self.endpoint = endpoint
        self.client = CommanderClient(endpoint)

        self.history = int(history)
        self.poll_dt_ms = int(max(10, round(1000.0 / max(0.1, poll_hz))))

        self.x = deque(maxlen=self.history)
        self.y = deque(maxlen=self.history)

        self._last_info = {}
        self._build_ui()
        self._refresh_info()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.poll_dt_ms)

    def _build_ui(self):
        layout = QVBoxLayout()

        # --- top controls ---
        top = QGridLayout()
        top.addWidget(QLabel(f"Beam {self.beam}"), 0, 0, 1, 2)

        self.field = QComboBox()
        self.field.setEditable(False)

        self.reducer = QComboBox()
        self.reducer.addItems(["mean", "rms", "index"])

        self.idx_spin = QSpinBox()
        self.idx_spin.setRange(0, 10_000_000)
        self.idx_spin.setValue(0)

        self.mode_2d = QCheckBox("2D")
        self.mode_2d.setChecked(False)

        self.mode_dm = QCheckBox("DM map")
        self.mode_dm.setChecked(False)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 2000)
        self.n_spin.setValue(200)

        self.dec_spin = QSpinBox()
        self.dec_spin.setRange(1, 128)
        self.dec_spin.setValue(1)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)

        self.refresh_btn = QPushButton("Refresh fields")
        self.refresh_btn.clicked.connect(self._refresh_info)

        top.addWidget(QLabel("Field"), 1, 0); top.addWidget(self.field, 1, 1)
        top.addWidget(QLabel("Reducer"), 2, 0); top.addWidget(self.reducer, 2, 1)
        top.addWidget(QLabel("Index"), 3, 0); top.addWidget(self.idx_spin, 3, 1)
        top.addWidget(QLabel("n"), 4, 0); top.addWidget(self.n_spin, 4, 1)
        top.addWidget(QLabel("decimate"), 5, 0); top.addWidget(self.dec_spin, 5, 1)
        top.addWidget(self.mode_2d, 6, 0)
        top.addWidget(self.mode_dm, 6, 1)
        top.addWidget(self.clear_btn, 7, 0)
        top.addWidget(self.refresh_btn, 7, 1)

        # --- plots ---
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot()

        self.img = pg.ImageView()
        self.img.setVisible(False)

        layout.addLayout(top)
        layout.addWidget(self.plot)
        layout.addWidget(self.img)

        # --- command line ---
        cmdrow = QHBoxLayout()
        self.cmd_in = QLineEdit()
        self.cmd_in.setPlaceholderText('e.g. status   |   poll_telem {"n":1,"fields":["t_s","opd_metric"]}')
        self.cmd_send = QPushButton("Send")
        self.cmd_send.clicked.connect(self._send_cmd)
        cmdrow.addWidget(self.cmd_in)
        cmdrow.addWidget(self.cmd_send)

        self.cmd_out = QTextEdit()
        self.cmd_out.setReadOnly(True)
        self.cmd_out.setMaximumHeight(120)

        layout.addLayout(cmdrow)
        layout.addWidget(self.cmd_out)

        self.setLayout(layout)

        self.mode_2d.stateChanged.connect(self._update_mode_visibility)
        self.mode_dm.stateChanged.connect(self._update_mode_visibility)
        self._update_mode_visibility()

    def _update_mode_visibility(self):
        use_img = self.mode_2d.isChecked() or self.mode_dm.isChecked()
        self.img.setVisible(use_img)
        self.plot.setVisible(not use_img)

    def _log(self, s: str):
        self.cmd_out.append(s)

    def _send_cmd(self):
        msg = self.cmd_in.text().strip()
        if not msg:
            return
        reply = self.client.send_raw(msg)
        self._log(f"> {msg}\n{reply}\n")
        self.cmd_in.clear()

    def _clear(self):
        self.x.clear()
        self.y.clear()
        self.curve.setData([], [])

    def _refresh_info(self):
        info = self.client.call("poll_telem_info")
        self._last_info = info if isinstance(info, dict) else {}
        # Field list: keep it explicit to your TelemetryChunk attributes
        # (This is the “safe set”; add more if you extend the ring)
        fields = [
            "t_s", "frame_id", "lo_state", "ho_state", "paused", "overruns",
            "opd_metric", "snr_metric",
            "i_raw", "i_space", "s",
            "e_lo", "e_ho", "u_lo", "u_ho",
            "c_lo", "c_ho", "cmd",
        ]
        self.field.clear()
        self.field.addItems(fields)
        # reasonable defaults
        self.field.setCurrentText("opd_metric")

        if info.get("ok", False):
            self._log(f"[beam{self.beam}] ring: cap={info.get('capacity')} count={info.get('count')} overruns={info.get('overruns')}")
        else:
            self._log(f"[beam{self.beam}] poll_telem_info failed: {info}")

    def _tick(self):
        fld = self.field.currentText().strip()
        if not fld:
            return

        n = int(self.n_spin.value())
        dec = int(self.dec_spin.value())

        req = {"n": n, "decimate": dec, "fields": ["t_s", fld]}
        resp = self.client.call("poll_telem", json.dumps(req))
        if not resp.get("ok", False):
            return

        payload = resp.get("fields", {})
        t_s = payload.get("t_s", None)
        v = payload.get(fld, None)

        if t_s is None or v is None:
            return

        # t_s is list
        t_s = np.asarray(t_s, dtype=float).ravel()
        if t_s.size == 0:
            return

        # scalar vs vector
        arr = np.asarray(v)

        # 2D / DM modes: render latest sample only
        if self.mode_dm.isChecked():
            # expect a DM-length vector per sample; take last sample vector
            # poll_telem returns nested lists for arrays: shape (N, n_act) -> list-of-lists
            vv = np.asarray(v, dtype=float)
            if vv.ndim == 2:
                last = vv[-1]
            else:
                last = vv.ravel()
            try:
                img = dm140_to_12x12(last)
            except Exception:
                img = np.full((12, 12), np.nan, dtype=float)
            self.img.setImage(img.T, autoLevels=True)  # transpose to make it look “right-ish”
            return

        if self.mode_2d.isChecked():
            vv = np.asarray(v, dtype=float)
            if vv.ndim == 2:
                last = vv[-1]
            else:
                last = vv.ravel()
            img = try_square_reshape(last)
            if img is None:
                # fallback: show as 1 x N strip
                img = last.reshape(1, -1)
            self.img.setImage(img.T, autoLevels=True)
            return

        # time-series mode: reduce vectors to scalars
        if arr.ndim == 1:
            # already scalar series
            y = arr.astype(float)
        elif arr.ndim == 2:
            # reduce each sample
            reducer = self.reducer.currentText()
            idx = int(self.idx_spin.value())
            y = np.array([reduce_vector(arr[i], reducer, idx) for i in range(arr.shape[0])], dtype=float)
        else:
            y = np.full(t_s.size, np.nan, dtype=float)

        # append to history
        for tt, yy in zip(t_s, y):
            self.x.append(float(tt))
            self.y.append(float(yy))

        if len(self.x) > 1:
            x0 = self.x[0]
            xs = np.array(self.x, dtype=float) - x0
            ys = np.array(self.y, dtype=float)
            self.curve.setData(xs, ys)


# ---------------- main window ----------------
class MainWindow(QMainWindow):
    def __init__(self, panels: list[BeamPanel]):
        super().__init__()
        self.setWindowTitle("Baldr RTC Telemetry GUI")
        w = QWidget()
        layout = QHBoxLayout()
        for p in panels:
            layout.addWidget(p)
        w.setLayout(layout)
        self.setCentralWidget(w)


def default_socket_for_beam(beam: int) -> str:
    # matches your python RTC default mapping (3001, 3002, ...)
    return f"tcp://127.0.0.1:300{int(beam)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", type=str, default="1", help="comma-separated beams, e.g. 1 or 1,2,3,4")
    ap.add_argument("--history", type=int, default=1200, help="samples of scalar history to keep in GUI")
    ap.add_argument("--poll-hz", type=float, default=10.0, help="poll rate (Hz)")
    ap.add_argument("--socket", type=str, default="", help="override single socket (only if one beam)")
    args = ap.parse_args()

    beams = [int(x) for x in args.beams.split(",") if x.strip()]
    if args.socket and len(beams) != 1:
        raise SystemExit("--socket only allowed with a single beam")

    app = QApplication(sys.argv)

    panels = []
    for b in beams:
        endpoint = args.socket if args.socket else default_socket_for_beam(b)
        panels.append(BeamPanel(beam=b, endpoint=endpoint, history=args.history, poll_hz=args.poll_hz))

    win = MainWindow(panels)
    win.resize(1400, 700)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()