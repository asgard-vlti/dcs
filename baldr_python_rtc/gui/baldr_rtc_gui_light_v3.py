#!/usr/bin/env python3
import sys
import json
import argparse
import zmq
from PyQt5 import QtWidgets

# ---------------- ZMQ helper ----------------
def send_cmd(sock, cmd, payload=None):
    if payload is None:
        msg = cmd
    else:
        msg = f"{cmd} {json.dumps(payload)}"
    sock.send_string(msg)
    return sock.recv_string()

# ---------------- GUI ----------------
class LightRTC(QtWidgets.QWidget):
    def __init__(self, beam, addr):
        super().__init__()
        self.beam = beam
        self.setWindowTitle(f"BALDR Light GUI – Beam {beam}")

        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.connect(addr)

        layout = QtWidgets.QVBoxLayout(self)

        # --- quick buttons
        btn_row = QtWidgets.QHBoxLayout()
        for label, cmd in [
            ("OPEN_LO", "open_baldr_LO"),
            ("OPEN_HO", "open_baldr_HO"),
            ("CLOSE_LO", "close_baldr_LO"),
            ("CLOSE_HO", "close_baldr_HO"),
            ("PHOT", "update_N0_runtime"),
            ("UPDATE ZWFS INT. SETPOINT", "update_I0_runtime"),
        ]:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _, c=cmd: self.run_cmd(c))
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # --- gain controls
        grid = QtWidgets.QGridLayout()
        row = 0
        for tag, space in [
            ("ki_LO", "lo"),
            ("rho_LO", "lo"),
            ("ki_HO", "ho"),
            ("rho_HO", "ho"),
        ]:
            grid.addWidget(QtWidgets.QLabel(tag), row, 0)

            idx = QtWidgets.QLineEdit("all")
            val = QtWidgets.QLineEdit("0.0")
            btn = QtWidgets.QPushButton("Update")

            def make_cb(param=tag.split("_")[0], sp=space, i=idx, v=val):
                return lambda: self.set_gain(sp, param, i.text(), float(v.text()))

            btn.clicked.connect(make_cb())

            grid.addWidget(QtWidgets.QLabel("indices"), row, 1)
            grid.addWidget(idx, row, 2)
            grid.addWidget(QtWidgets.QLabel("value"), row, 3)
            grid.addWidget(val, row, 4)
            grid.addWidget(btn, row, 5)
            row += 1

        layout.addLayout(grid)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def run_cmd(self, cmd):
        try:
            rep = send_cmd(self.sock, cmd)
            self.log.append(f"> {cmd}\n{rep}\n")
        except Exception as e:
            self.log.append(f"ERROR {cmd}: {e}\n")

    def set_gain(self, space, param, indices, value):
        cmd = f"set_{space}_gain"
        payload = [param, indices.replace(" ", ""), value]
        try:
            rep = send_cmd(self.sock, cmd, payload)
            self.log.append(f"> {cmd} {payload}\n{rep}\n")
        except Exception as e:
            self.log.append(f"ERROR {cmd}: {e}\n")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, required=True)
    ap.add_argument("--addr", type=str, default="tcp://127.0.0.1:5555")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    w = LightRTC(args.beam, args.addr)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
