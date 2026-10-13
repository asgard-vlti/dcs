"""Compare bias-subtracted pupil flux while scanning DM sparkle separation.

Run on mimir in the asg environment with::

    python -m minimal_baldr_python_rtc.experiments.cold_stop_sparkle_scan 1

The scan uses 45 degrees and reaches 6.5 lambda/D. The selected beam's disp03
channel is cleared at the start and end of each scan.
Do not run another sparkle or disp03 controller for that beam at the same time.
"""

import argparse
import threading
import time

import numpy as onp
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from dcs.cmd_scripts.sparkles import SparklesController


SCAN_ANGLE_DEG = 45.0
SEPARATIONS = (0.0,) + tuple(round(index / 10, 1) for index in range(1, 66))
SETTLE_S = 0.02
DISCARD_FRAMES = 2
AVERAGE_FRAMES = 16
PIXEL_BIAS_COUNTS = 1000.0
FRAME_WAIT_S = 0.2
CAMERA_STALL_S = 2.0


class ScanCancelled(Exception):
    """The window closed during a scan."""


class CameraReader:
    """Read distinct, stable frames from the processed Baldr camera stream."""

    def __init__(self, beam, shm_factory=None):
        if shm_factory is None:
            from xaosim.shmlib import shm

            shm_factory = shm
        path = f"/dev/shm/baldr{beam}.im.shm"
        self.stream = shm_factory(path, nosem=True)
        if getattr(self.stream, "empty", False):
            raise FileNotFoundError(f"Missing Baldr camera stream: {path}")
        try:
            self.counter = self.stream.get_counter()
        except Exception:
            self.close()
            raise

    def read_fresh(self, cancelled):
        deadline = time.monotonic() + CAMERA_STALL_S
        while True:
            if cancelled.is_set():
                raise ScanCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Baldr camera produced no fresh frame for 2 seconds")

            frame = self.stream.get_data(
                check=self.counter, timeout=min(FRAME_WAIT_S, remaining)
            )
            counter = self.stream.get_counter()
            if counter < self.counter:
                raise RuntimeError("Baldr camera counter reset during scan")
            if counter == self.counter:
                continue

            frame = onp.asarray(frame).copy()
            self.counter = counter
            if self.stream.get_counter() != counter:
                continue
            if frame.ndim != 2:
                raise ValueError(f"Expected a 2D Baldr frame, got {frame.shape}")
            return frame

    def close(self):
        self.stream.close(erase_file=False)


def acquire_scan(
    beam,
    amplitude,
    cancelled=None,
    controller_factory=SparklesController,
    camera_factory=CameraReader,
    sleep_fn=time.sleep,
    separations=SEPARATIONS,
    on_progress=None,
):
    """Return separation and mean whole-frame flux arrays for one scan."""
    cancelled = cancelled if cancelled is not None else threading.Event()
    controller = None
    camera = None
    fluxes = []
    try:
        controller = controller_factory(beam)
        camera = camera_factory(beam)
        for index, separation in enumerate(separations):
            if cancelled.is_set():
                raise ScanCancelled()
            controller.set_enabled(
                separation != 0.0, separation or 0.1, SCAN_ANGLE_DEG, amplitude
            )
            sleep_fn(SETTLE_S)
            if cancelled.is_set():
                raise ScanCancelled()
            for _ in range(DISCARD_FRAMES):
                camera.read_fresh(cancelled)
            total = 0.0
            for _ in range(AVERAGE_FRAMES):
                frame = camera.read_fresh(cancelled)
                flux = float(
                    onp.sum(frame, dtype=onp.float64)
                    - PIXEL_BIAS_COUNTS * frame.size
                )
                if not onp.isfinite(flux):
                    raise ValueError("Baldr camera returned non-finite flux")
                total += flux
            fluxes.append(total / AVERAGE_FRAMES)
            if on_progress is not None:
                on_progress(index + 1, len(separations))
        return onp.asarray(separations), onp.asarray(fluxes)
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            if controller is not None:
                controller.close()


class ScanWorker(QtCore.QObject):
    completed = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int, int)
    finished = QtCore.pyqtSignal()

    def __init__(self, beam, amplitude):
        super().__init__()
        self.beam = beam
        self.amplitude = amplitude
        self.cancelled = threading.Event()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            separations, fluxes = acquire_scan(
                self.beam,
                self.amplitude,
                cancelled=self.cancelled,
                on_progress=self.progress.emit,
            )
        except ScanCancelled:
            self.failed.emit("Scan cancelled")
        except Exception as exc:
            self.failed.emit(f"Scan failed: {exc}")
        else:
            self.completed.emit(separations, fluxes)
        finally:
            self.finished.emit()


class ScanWindow(QtWidgets.QWidget):
    """Keep completed flux curves visible while the operator adjusts motors."""

    def __init__(self, beam=1):
        super().__init__()
        self.setWindowTitle("Cold-stop sparkle flux scan")
        self._thread = None
        self._worker = None
        self._closing = False
        self._run_number = 0
        self._curves = []
        self._active_parameters = None

        self.beam_input = QtWidgets.QComboBox()
        for number in range(1, 5):
            self.beam_input.addItem(f"Beam {number}", number)
        self.beam_input.setCurrentIndex(beam - 1)

        self.angle_label = QtWidgets.QLabel(f"{SCAN_ANGLE_DEG:g}° (fixed)")

        self.amplitude_input = QtWidgets.QDoubleSpinBox()
        self.amplitude_input.setRange(0.0, 0.2)
        self.amplitude_input.setDecimals(3)
        self.amplitude_input.setSingleStep(0.005)
        self.amplitude_input.setValue(0.02)

        self.run_button = QtWidgets.QPushButton("Run")
        self.clear_button = QtWidgets.QPushButton("Clear Lines")
        self.status = QtWidgets.QLabel("Ready — place the beam on the camera in pupil mode")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Sparkle separation", units="λ/D")
        self.plot_widget.setLabel(
            "left", "Mean bias-subtracted flux", units="counts/frame"
        )
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setXRange(0.0, 6.5, padding=0.02)
        self.legend = self.plot_widget.addLegend()

        controls = QtWidgets.QFormLayout()
        controls.addRow("Beam", self.beam_input)
        controls.addRow("Angle from DM horizontal", self.angle_label)
        controls.addRow("Amplitude (DM units)", self.amplitude_input)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.clear_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(buttons)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.status)
        self.resize(850, 580)

        self.run_button.clicked.connect(self._start_scan)
        self.clear_button.clicked.connect(self._clear_lines)

    def _set_running(self, running):
        for widget in (
            self.beam_input,
            self.amplitude_input,
            self.run_button,
            self.clear_button,
        ):
            widget.setEnabled(not running)

    def _start_scan(self):
        if self._thread is not None:
            return
        parameters = (
            self.beam_input.currentData(),
            self.amplitude_input.value(),
        )
        self._active_parameters = parameters
        self._thread = QtCore.QThread(self)
        self._worker = ScanWorker(*parameters)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._show_progress)
        self._worker.completed.connect(self._add_run)
        self._worker.failed.connect(self.status.setText)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._scan_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._set_running(True)
        self.status.setText(f"Scanning: 0/{len(SEPARATIONS)} separations")
        self._thread.start()

    def _show_progress(self, completed, total):
        if not self._closing:
            self.status.setText(f"Scanning: {completed}/{total} separations")

    def _add_run(self, separations, fluxes):
        if self._closing:
            return
        beam, amplitude = self._active_parameters
        color = QtGui.QColor.fromHsvF((self._run_number * 0.618034) % 1.0, 0.75, 0.95)
        name = f"Run {self._run_number + 1} (B{beam}, {SCAN_ANGLE_DEG:g}°, {amplitude:.3f})"
        curve = self.plot_widget.plot(
            separations,
            fluxes,
            pen=pg.mkPen(color, width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=color,
            name=name,
        )
        self._curves.append(curve)
        self._run_number += 1
        self.status.setText(f"Completed {name}")

    def _clear_lines(self):
        for curve in self._curves:
            self.plot_widget.removeItem(curve)
        self.legend.clear()
        self._curves.clear()
        self._run_number = 0
        self.status.setText("Lines cleared")

    def _scan_finished(self):
        self._thread = None
        self._worker = None
        self._active_parameters = None
        if not self._closing:
            self._set_running(False)

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._closing = True
            self._worker.cancelled.set()
            self._thread.quit()
            if not self._thread.wait(3000):
                self._closing = False
                self.status.setText("Waiting for scan to stop and clear disp03")
                event.ignore()
                return
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("beam", nargs="?", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = ScanWindow(args.beam)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
