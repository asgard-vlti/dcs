"""Apply a static four-spot Fourier pattern to one ASGARD DM."""

import argparse
from pathlib import Path

import hcipy
import numpy as onp
from PyQt5 import QtWidgets

from minimal_baldr_python_rtc.basis_funcs import make_hc_act_grid


DM_SIZE = 12
CHANNEL = 3
ZERO_MAP = onp.zeros((DM_SIZE, DM_SIZE), dtype=onp.float64)
ACTIVE = onp.ones((DM_SIZE, DM_SIZE), dtype=bool)
ACTIVE[0, 0] = ACTIVE[0, -1] = ACTIVE[-1, 0] = ACTIVE[-1, -1] = False


def make_sparkle_map(distance: float, angle: float, amplitude: float) -> onp.ndarray:
    """Return two orthogonal Fourier modes in normalized DM command units."""
    values = (distance, angle, amplitude)
    if not all(onp.isfinite(value) for value in values):
        raise ValueError("Distance, angle, and amplitude must be finite")
    max_distance = 6.5 if angle == 45.0 else 5.0
    if not 0.1 <= distance <= max_distance:
        raise ValueError(f"Distance must be 0.1–{max_distance:.1f} lambda/D")
    if not 0.0 <= angle <= 90.0:
        raise ValueError("Angle must be 0–90 degrees")
    if not 0.0 <= amplitude <= 0.2:
        raise ValueError("Amplitude must be 0–0.2")
    if amplitude == 0:
        return ZERO_MAP.copy()

    grid = make_hc_act_grid()
    theta = onp.deg2rad(angle)
    x = onp.asarray(grid.x)
    y = onp.asarray(grid.y)
    first = distance * (onp.cos(theta) * x + onp.sin(theta) * y)
    second = distance * (-onp.sin(theta) * x + onp.cos(theta) * y)

    # A quarter-cycle phase keeps cardinal modes nonzero at the 5 lambda/D Nyquist limit.
    phase = onp.pi / 4
    mode_a = hcipy.Field(onp.cos(2 * onp.pi * first + phase), grid)
    mode_b = hcipy.Field(onp.cos(2 * onp.pi * second + phase), grid)
    pattern = onp.asarray(mode_a + mode_b).reshape(DM_SIZE, DM_SIZE).copy()

    pattern[0, :] = pattern[1, :]
    pattern[-1, :] = pattern[-2, :]
    pattern[:, 0] = pattern[:, 1]
    pattern[:, -1] = pattern[:, -2]
    pattern[~ACTIVE] = 0.0
    pattern[ACTIVE] -= pattern[ACTIVE].mean()

    peak = onp.max(onp.abs(pattern[ACTIVE]))
    if peak < 1e-12:
        raise ValueError("This distance and angle produce a degenerate DM pattern")
    pattern[ACTIVE] *= amplitude / peak
    return pattern


class DmChannel:
    """Own the additive disp03 map for one beam."""

    def __init__(self, beam: int):
        if beam not in (1, 2, 3, 4):
            raise ValueError("Beam must be 1–4")

        channel_path = Path(f"/dev/shm/dm{beam}disp{CHANNEL:02d}.im.shm")
        combined_path = Path(f"/dev/shm/dm{beam}.im.shm")
        for path in (channel_path, combined_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing DM shared memory: {path}")

        from xaosim.shmlib import shm

        self.channel = None
        self.combined = None
        try:
            self.channel = shm(str(channel_path), nosem=False)
            self.combined = shm(str(combined_path), nosem=False)
            if self.channel.empty or self.combined.empty:
                raise RuntimeError(f"Could not open DM shared memory for beam {beam}")
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def write(self, pattern: onp.ndarray) -> None:
        array = onp.asarray(pattern, dtype=onp.float64)
        if array.shape != (DM_SIZE, DM_SIZE) or not onp.all(onp.isfinite(array)):
            raise ValueError("DM pattern must be a finite 12×12 map")
        self.channel.set_data(array)
        self.combined.post_sems(1)

    def clear(self) -> None:
        self.write(ZERO_MAP)

    def close(self) -> None:
        errors = []
        for stream in (self.channel, self.combined):
            if stream is not None and not stream.empty:
                try:
                    stream.close(erase_file=False)
                except Exception as exc:
                    errors.append(exc)
        self.channel = None
        self.combined = None
        if errors:
            raise RuntimeError(f"Could not close DM shared memory: {errors[0]}") from errors[0]


class SparklesController:
    """Apply and clear the selected beam's static sparkle pattern."""

    def __init__(self, beam: int = 1, channel_factory=DmChannel):
        self.channel_factory = channel_factory
        self.channel = channel_factory(beam)
        self.beam = beam
        self.enabled = False
        try:
            self.channel.clear()
        except Exception:
            self.channel.close()
            raise

    def set_enabled(self, enabled: bool, distance: float, angle: float, amplitude: float) -> None:
        if enabled:
            self.channel.write(make_sparkle_map(distance, angle, amplitude))
        else:
            self.channel.clear()
        self.enabled = enabled

    def update(self, distance: float, angle: float, amplitude: float) -> None:
        if self.enabled:
            self.channel.write(make_sparkle_map(distance, angle, amplitude))

    def set_beam(self, beam: int, distance: float, angle: float, amplitude: float) -> None:
        if beam == self.beam:
            return
        new_channel = self.channel_factory(beam)
        try:
            self.channel.clear()
        except Exception:
            new_channel.close()
            raise

        self.channel.close()
        self.channel = new_channel
        self.beam = beam
        if self.enabled:
            self.channel.write(make_sparkle_map(distance, angle, amplitude))
        else:
            self.channel.clear()

    def close(self) -> None:
        self.channel.clear()
        self.channel.close()


class SparklesWindow(QtWidgets.QWidget):
    """Small control panel for the selected beam's disp03 map."""

    def __init__(self, beam: int = 1, channel_factory=DmChannel):
        super().__init__()
        self.controller = SparklesController(beam, channel_factory)
        self.setWindowTitle("DM Sparkles")

        layout = QtWidgets.QFormLayout(self)
        self.beam_input = QtWidgets.QComboBox()
        for beam_number in range(1, 5):
            self.beam_input.addItem(f"Beam {beam_number}", beam_number)
        self.beam_input.setCurrentIndex(beam - 1)

        self.angle_input = QtWidgets.QDoubleSpinBox()
        self.angle_input.setRange(0, 90)
        self.angle_input.setDecimals(1)
        self.angle_input.setSingleStep(0.1)
        self.angle_input.setSuffix("°")

        self.distance_input = QtWidgets.QDoubleSpinBox()
        self.distance_input.setRange(0.1, 5.0)
        self.distance_input.setDecimals(3)
        self.distance_input.setSingleStep(0.1)
        self.distance_input.setValue(3.0)

        self.amplitude_input = QtWidgets.QDoubleSpinBox()
        self.amplitude_input.setRange(0.0, 0.2)
        self.amplitude_input.setDecimals(3)
        self.amplitude_input.setSingleStep(0.005)
        self.amplitude_input.setValue(0.02)

        self.on_input = QtWidgets.QCheckBox("On")
        self.status = QtWidgets.QLabel("Off — disp03 cleared")

        layout.addRow("Beam", self.beam_input)
        layout.addRow("Angle from DM horizontal", self.angle_input)
        layout.addRow("Distance (λ/D)", self.distance_input)
        layout.addRow("Amplitude (DM units)", self.amplitude_input)
        layout.addRow("Sparkles", self.on_input)
        layout.addRow(self.status)

        self.beam_input.currentIndexChanged.connect(self._change_beam)
        self.angle_input.valueChanged.connect(self._update)
        self.distance_input.valueChanged.connect(self._update)
        self.amplitude_input.valueChanged.connect(self._update)
        self.on_input.toggled.connect(self._toggle)

    def _parameters(self) -> tuple[float, float, float]:
        return (
            self.distance_input.value(),
            self.angle_input.value(),
            self.amplitude_input.value(),
        )

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status.setText(message)
        self.status.setStyleSheet("color: #b00020;" if error else "")

    def _toggle(self, checked: bool) -> None:
        try:
            self.controller.set_enabled(checked, *self._parameters())
        except Exception as exc:
            self.on_input.blockSignals(True)
            self.on_input.setChecked(self.controller.enabled)
            self.on_input.blockSignals(False)
            self._set_status(f"DM write failed: {exc}", error=True)
            return
        self._set_status("On — disp03 active" if checked else "Off — disp03 cleared")

    def _update(self, _value: float) -> None:
        try:
            self.controller.update(*self._parameters())
        except Exception as exc:
            self._set_status(f"DM update failed: {exc}", error=True)
            return
        if self.controller.enabled:
            self._set_status("On — disp03 active")

    def _change_beam(self, _index: int) -> None:
        beam = self.beam_input.currentData()
        try:
            self.controller.set_beam(beam, *self._parameters())
        except Exception as exc:
            if self.controller.beam != beam:
                self.beam_input.blockSignals(True)
                self.beam_input.setCurrentIndex(self.controller.beam - 1)
                self.beam_input.blockSignals(False)
            self._set_status(f"Beam change failed: {exc}", error=True)
            return
        self._set_status(
            "On — disp03 active" if self.controller.enabled else "Off — disp03 cleared"
        )

    def closeEvent(self, event) -> None:
        try:
            self.controller.close()
        except Exception as exc:
            event.ignore()
            self._set_status(f"Could not clear disp03: {exc}", error=True)
            return
        event.accept()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Static DM sparkles control panel")
    parser.add_argument("beam", nargs="?", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    try:
        window = SparklesWindow(args.beam)
    except Exception as exc:
        QtWidgets.QMessageBox.critical(None, "DM Sparkles", str(exc))
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
