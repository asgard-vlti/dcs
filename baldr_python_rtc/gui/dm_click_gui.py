#!/usr/bin/env python3
"""
Interactive DM click GUI for Baldr/ZWFS calibration products.

Behavior
--------
- Reads an existing Baldr TOML file.
- Uses stored I2A only to infer actuator positions if explicit coordinates are absent.
- Static mode uses a fixed random 32x32 image.
- Clicking in the image selects the nearest actuator in image pixel space.
- Clicking directly on an actuator dot selects that actuator exactly.
- The amplitude slider edits the currently selected actuator only.
- Previously edited actuators retain their amplitudes by default.
- Preview command = flat_cmd + actuator_offsets.
- No recalibration is performed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
import toml

try:
    from PyQt5 import QtCore, QtWidgets
except Exception:
    from PySide6 import QtCore, QtWidgets


def _import_live_backends():
    from xaosim.shmlib import shm  # type: ignore
    from asgard_alignment.DM_shm_ctrl import dmclass  # type: ignore
    return shm, dmclass


@dataclass
class CalibrationData:
    beam_id: int
    phasemask: str
    image_shape: tuple[int, int]
    i2a: np.ndarray
    actuator_xy_pix: np.ndarray
    flat_cmd: np.ndarray
    reference_image: Optional[np.ndarray]
    m2c: Optional[np.ndarray]


class TomlCalibrationLoader:
    def __init__(self, toml_path: str, beam_id: int, phasemask: str):
        self.toml_path = Path(toml_path)
        self.beam_id = int(beam_id)
        self.beam_key = f"beam{self.beam_id}"
        self.phasemask = phasemask

    def load(self) -> CalibrationData:
        if not self.toml_path.exists():
            raise FileNotFoundError(f"TOML file not found: {self.toml_path}")

        data = toml.load(self.toml_path)
        if self.beam_key not in data:
            raise KeyError(f"Missing TOML section '{self.beam_key}' in {self.toml_path}")

        beam_data = data[self.beam_key]
        i2a = self._load_i2a(beam_data)
        ny, nx = self._infer_image_shape(i2a, beam_data)

        if i2a.shape[1] != ny * nx:
            raise ValueError(
                f"I2A has n_pix={i2a.shape[1]}, but inferred image shape is {(ny, nx)} -> {ny * nx} pixels"
            )

        reference_image = self._load_reference_image(beam_data, (ny, nx))
        flat_cmd = self._load_flat_command(beam_data)
        m2c = self._load_m2c(beam_data)
        actuator_xy_pix = self._load_or_derive_actuator_positions(beam_data, i2a, (ny, nx))

        n_act = i2a.shape[0]
        if flat_cmd.size != n_act:
            if flat_cmd.size == 140 and n_act != 140:
                raise ValueError(f"Flat command has length 140 but I2A has {n_act} actuators.")
            if flat_cmd.size != 140 and n_act == 140:
                flat_cmd = np.zeros(140, dtype=float)
            elif flat_cmd.size != n_act:
                flat_cmd = np.zeros(n_act, dtype=float)

        return CalibrationData(
            beam_id=self.beam_id,
            phasemask=self.phasemask,
            image_shape=(ny, nx),
            i2a=i2a,
            actuator_xy_pix=actuator_xy_pix,
            flat_cmd=flat_cmd,
            reference_image=reference_image,
            m2c=m2c,
        )

    def _load_i2a(self, beam_data: dict) -> np.ndarray:
        if "I2A" not in beam_data:
            raise KeyError(f"Missing '{self.beam_key}.I2A' in TOML.")
        i2a = np.asarray(beam_data["I2A"], dtype=float)
        if i2a.ndim != 2:
            raise ValueError(f"Expected I2A to be 2D, got shape {i2a.shape}")
        return i2a

    def _infer_image_shape(self, i2a: np.ndarray, beam_data: dict) -> tuple[int, int]:
        n_pix = int(i2a.shape[1])
        n_side = int(round(np.sqrt(n_pix)))

        ctrl_model = (
            beam_data.get(self.phasemask, {}).get("ctrl_model", {})
            if isinstance(beam_data.get(self.phasemask, {}), dict)
            else {}
        )

        for key in ("N0", "I0", "norm_pupil", "dark"):
            if key in ctrl_model:
                arr = np.asarray(ctrl_model[key], dtype=float).reshape(-1)
                if arr.size == n_pix:
                    return (n_side, n_side)

        if n_side * n_side == n_pix:
            return (n_side, n_side)

        raise ValueError("Could not infer image shape from TOML.")

    def _load_reference_image(
        self,
        beam_data: dict,
        image_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        ctrl_model = beam_data.get(self.phasemask, {}).get("ctrl_model", {})
        for key in ("N0", "I0", "norm_pupil", "dark"):
            if key in ctrl_model:
                arr = np.asarray(ctrl_model[key], dtype=float).reshape(-1)
                if arr.size == image_shape[0] * image_shape[1]:
                    return arr.reshape(image_shape)
        return None

    def _load_flat_command(self, beam_data: dict) -> np.ndarray:
        flat_cmd = np.zeros(140, dtype=float)

        possible_paths = [
            beam_data.get("dm_flat", None),
            beam_data.get("DM_flat", None),
            beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("dm_flat", None),
            beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("DM_flat", None),
        ]
        for entry in possible_paths:
            if entry is None:
                continue
            if isinstance(entry, (str, bytes)):
                continue
            try:
                arr = np.asarray(entry, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if arr.size == 140:
                flat_cmd = arr.copy()
                break
            if arr.size == 144:
                flat_cmd = self._map_144_to_140(arr)
                break

        return flat_cmd

    def _load_m2c(self, beam_data: dict) -> Optional[np.ndarray]:
        ctrl_model = beam_data.get(self.phasemask, {}).get("ctrl_model", {})
        if "M2C" not in ctrl_model:
            return None
        m2c = np.asarray(ctrl_model["M2C"], dtype=float)
        if m2c.ndim != 2:
            return None
        return m2c

    def _load_or_derive_actuator_positions(
        self,
        beam_data: dict,
        i2a: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        explicit_keys = [
            "actuator_coord_list_pixel_space",
            "actuator_coords_pixel_space",
            "actuator_pixel_coords",
        ]
        for key in explicit_keys:
            if key in beam_data:
                arr = np.asarray(beam_data[key], dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] == i2a.shape[0]:
                    return arr

        return self._derive_positions_from_i2a(i2a, image_shape)

    @staticmethod
    def _derive_positions_from_i2a(i2a: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        ny, nx = image_shape
        yy, xx = np.indices((ny, nx), dtype=float)
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        coords = np.zeros((i2a.shape[0], 2), dtype=float)
        for k, row in enumerate(i2a):
            weights = np.abs(np.asarray(row, dtype=float).reshape(-1))
            total = float(np.sum(weights))
            if not np.isfinite(total) or total <= 0:
                peak = int(np.argmax(np.abs(row)))
                coords[k, 0] = xx_flat[peak]
                coords[k, 1] = yy_flat[peak]
                continue
            coords[k, 0] = np.sum(weights * xx_flat) / total
            coords[k, 1] = np.sum(weights * yy_flat) / total
        return coords

    @staticmethod
    def _map_144_to_140(cmd144: np.ndarray) -> np.ndarray:
        grid = np.asarray(cmd144, dtype=float).reshape(12, 12)
        keep = np.ones((12, 12), dtype=bool)
        keep[0, 0] = False
        keep[0, -1] = False
        keep[-1, 0] = False
        keep[-1, -1] = False
        return grid[keep].reshape(-1)


class ImageSourceBase(QtCore.QObject):
    image_ready = QtCore.pyqtSignal(object)

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class StaticImageSource(ImageSourceBase):
    def __init__(self, shape: tuple[int, int], seed: int = 1234):
        super().__init__()
        rng = np.random.default_rng(seed)
        self._image = rng.normal(size=shape).astype(float)

    def start(self) -> None:
        self.image_ready.emit(self._image.copy())

    def stop(self) -> None:
        pass



class LiveShmImageSource(ImageSourceBase):
    def __init__(self, beam_id: int, timer_ms: int = 300):
        super().__init__()
        shm, _ = _import_live_backends()
        self._shm = shm(f"/dev/shm/baldr{beam_id}.im.shm")
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(timer_ms))
        self._timer.timeout.connect(self._poll)

    def _poll(self) -> None:
        try:
            img = np.asarray(self._shm.get_data(), dtype=float)
            self.image_ready.emit(img)
        except Exception as exc:
            print(f"Warning: failed to read SHM image: {exc}", file=sys.stderr)

    def start(self) -> None:
        self._poll()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        try:
            self._shm.close(erase_file=False)
        except Exception:
            pass

# also we can try average frames something like this , but need some sleeps 
# class LiveShmImageSource(ImageSourceBase):
#     def __init__(self, beam_id: int, timer_ms: int = 1000, n_average: int = 4):
#         super().__init__()
#         shm, _ = _import_live_backends()
#         self._shm = shm(f"/dev/shm/baldr{beam_id}.im.shm")
#         self._timer = QtCore.QTimer(self)
#         self._timer.setInterval(int(timer_ms))
#         self._timer.timeout.connect(self._poll)
#         self._n_average = int(max(1, n_average))

#     def _poll(self) -> None:
#         try:
#             frames = []
#             for _ in range(self._n_average):
#                 frames.append(np.asarray(self._shm.get_data(), dtype=float))
#             img = np.mean(frames, axis=0)
#             self.image_ready.emit(img)
#         except Exception as exc:
#             print(f"Warning: failed to read SHM image: {exc}", file=sys.stderr)

#     def start(self) -> None:
#         self._poll()
#         self._timer.start()

#     def stop(self) -> None:
#         self._timer.stop()
#         try:
#             self._shm.close(erase_file=False)
#         except Exception:
#             pass


class DmSinkBase:
    def send(self, cmd: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullDmSink(DmSinkBase):
    def send(self, cmd: np.ndarray) -> None:
        return


class LiveDmSink(DmSinkBase):
    def __init__(self, beam_id: int):
        _, dmclass = _import_live_backends()
        self._dm = dmclass(beam_id=beam_id)

    def send(self, cmd: np.ndarray) -> None:
        cmd = np.asarray(cmd, dtype=float).reshape(-1)
        if cmd.size != 140:
            raise ValueError(f"Expected 140-length DM command, got {cmd.size}")
        cmd2d = self._dm.cmd_2_map2D(cmd, fill=0)
        # self._dm.set_data(cmd2d)
        print(cmd.shape)
        self._dm.set_data_chn(cmd, chn=3)


@dataclass
class ControllerState:
    click_row: Optional[int] = None
    click_col: Optional[int] = None
    selected_actuator: Optional[int] = None
    actuator_offsets: np.ndarray = field(default_factory=lambda: np.zeros(140, dtype=float))
    preview_cmd: Optional[np.ndarray] = None


class DmClickController(QtCore.QObject):
    state_changed = QtCore.pyqtSignal()

    def __init__(self, calib: CalibrationData, dm_sink: DmSinkBase):
        super().__init__()
        self.calib = calib
        self.dm_sink = dm_sink
        self.send_enabled = False

        n_act = calib.i2a.shape[0]
        self.state = ControllerState(actuator_offsets=np.zeros(n_act, dtype=float))
        self._refresh_preview()

    def set_click(self, row: int, col: int) -> None:
        ny, nx = self.calib.image_shape
        row = int(np.clip(row, 0, ny - 1))
        col = int(np.clip(col, 0, nx - 1))

        xy = self.calib.actuator_xy_pix
        d2 = (xy[:, 0] - col) ** 2 + (xy[:, 1] - row) ** 2
        nearest = int(np.argmin(d2))

        self.state.click_row = row
        self.state.click_col = col
        self.state.selected_actuator = nearest
        self.state_changed.emit()

    def set_selected_actuator(self, actuator_index: int) -> None:
        actuator_index = int(actuator_index)
        if not (0 <= actuator_index < len(self.state.actuator_offsets)):
            return
        self.state.selected_actuator = actuator_index
        x, y = self.calib.actuator_xy_pix[actuator_index]
        self.state.click_col = int(round(float(x)))
        self.state.click_row = int(round(float(y)))
        self.state_changed.emit()

    def set_selected_amplitude(self, value: float) -> None:
        if self.state.selected_actuator is None:
            return
        self.state.actuator_offsets[self.state.selected_actuator] = float(value)
        self._refresh_preview()

    def get_selected_amplitude(self) -> float:
        if self.state.selected_actuator is None:
            return 0.0
        return float(self.state.actuator_offsets[self.state.selected_actuator])

    def clear_selected_actuator(self) -> None:
        if self.state.selected_actuator is None:
            return
        self.state.actuator_offsets[self.state.selected_actuator] = 0.0
        self._refresh_preview()

    def clear_all_offsets(self) -> None:
        self.state.actuator_offsets[:] = 0.0
        self._refresh_preview()

    def zero_and_send(self) -> None:
        self.state.actuator_offsets[:] = 0.0
        self.state.preview_cmd = self.calib.flat_cmd.copy()
        if self.send_enabled:
            self.dm_sink.send(self.state.preview_cmd)
        self.state_changed.emit()

    def _refresh_preview(self) -> None:
        self.state.preview_cmd = self.calib.flat_cmd + self.state.actuator_offsets
        if self.send_enabled:
            self.dm_sink.send(self.state.preview_cmd)
        self.state_changed.emit()

    def top_contributors(self, n: int = 10) -> list[tuple[int, float]]:
        w = np.asarray(self.state.actuator_offsets, dtype=float).reshape(-1)
        mask = np.abs(w) > 0
        if not np.any(mask):
            return []
        order = np.argsort(np.abs(w))[::-1]
        out = []
        for i in order:
            if np.abs(w[i]) <= 0:
                continue
            out.append((int(i), float(w[i])))
            if len(out) >= n:
                break
        return out


class ClickableImageItem(pg.ImageItem):
    clicked = QtCore.pyqtSignal(float, float)

    def mouseClickEvent(self, ev):  # type: ignore[override]
        if ev.button() == QtCore.Qt.LeftButton:
            pos = ev.pos()
            self.clicked.emit(float(pos.x()), float(pos.y()))
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        calib: CalibrationData,
        image_source: ImageSourceBase,
        controller: DmClickController,
        mode: str,
        allow_send: bool,
    ):
        super().__init__()
        self.calib = calib
        self.image_source = image_source
        self.controller = controller
        self.mode = mode
        self.allow_send = allow_send

        self.dm_display_limit = 0.1
        self._ignore_slider_signal = False

        self.setWindowTitle(f"DM Click GUI | beam {calib.beam_id} | {mode}")
        self.resize(1180, 760)

        self._build_ui()
        self._connect_signals()

        self._render_actuator_overlay()
        self._update_dm_preview()
        self._update_status()

        self.image_source.start()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.graphics = pg.GraphicsLayoutWidget()
        self.view = self.graphics.addViewBox(lockAspect=True, invertY=False)
        self.view.setMouseEnabled(x=False, y=False)

        self.image_item = ClickableImageItem(axisOrder="row-major")
        self.view.addItem(self.image_item)

        self.scatter_all = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen("c", width=3.0),
            brush=None,
            pxMode=True,
        )
        self.scatter_selected = pg.ScatterPlotItem(
            size=18,
            pen=pg.mkPen("y", width=7),
            brush=None,
            pxMode=True,
        )
        self.scatter_active = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen("r", width=2),
            brush=None,
            pxMode=True,
        )
        self.click_marker = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen("w", width=2),
            brush=None,
            symbol="x",
            pxMode=True,
        )

        self.view.addItem(self.scatter_all)
        self.view.addItem(self.scatter_active)
        self.view.addItem(self.scatter_selected)
        self.view.addItem(self.click_marker)

        right = QtWidgets.QWidget()
        right.setMaximumWidth(380)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        mode_box = QtWidgets.QGroupBox("Mode")
        mode_layout = QtWidgets.QFormLayout(mode_box)
        self.mode_label = QtWidgets.QLabel(self.mode)
        self.beam_label = QtWidgets.QLabel(str(self.calib.beam_id))
        self.shape_label = QtWidgets.QLabel(f"{self.calib.image_shape[0]} x {self.calib.image_shape[1]}")
        mode_layout.addRow("Mode:", self.mode_label)
        mode_layout.addRow("Beam:", self.beam_label)
        mode_layout.addRow("Image shape:", self.shape_label)
        right_layout.addWidget(mode_box)

        amp_box = QtWidgets.QGroupBox("Selected actuator amplitude")
        amp_layout = QtWidgets.QVBoxLayout(amp_box)
        self.amp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.amp_slider.setRange(-1000, 1000)
        self.amp_slider.setValue(0)
        self.amp_slider.setTickInterval(100)
        self.amp_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.amp_value_label = QtWidgets.QLabel("0.0000")
        amp_layout.addWidget(self.amp_slider)
        amp_layout.addWidget(self.amp_value_label)
        right_layout.addWidget(amp_box)

        send_box = QtWidgets.QGroupBox("Commanding")
        send_layout = QtWidgets.QVBoxLayout(send_box)
        self.send_checkbox = QtWidgets.QCheckBox("Enable live send")
        self.send_checkbox.setChecked(False)
        self.send_checkbox.setEnabled(self.mode == "live" and self.allow_send)

        self.send_hint = QtWidgets.QLabel()
        self.clear_selected_button = QtWidgets.QPushButton("Clear selected actuator")
        self.clear_all_button = QtWidgets.QPushButton("Clear all offsets")
        self.zero_button = QtWidgets.QPushButton("Zero / send flat")

        send_layout.addWidget(self.send_checkbox)
        send_layout.addWidget(self.send_hint)
        send_layout.addWidget(self.clear_selected_button)
        send_layout.addWidget(self.clear_all_button)
        send_layout.addWidget(self.zero_button)
        right_layout.addWidget(send_box)

        click_box = QtWidgets.QGroupBox("Selection diagnostics")
        click_layout = QtWidgets.QFormLayout(click_box)
        self.pixel_label = QtWidgets.QLabel("-")
        self.index_label = QtWidgets.QLabel("-")
        self.selected_act_label = QtWidgets.QLabel("-")
        self.selected_amp_label = QtWidgets.QLabel("-")
        self.n_active_label = QtWidgets.QLabel("0")
        click_layout.addRow("Pixel (row, col):", self.pixel_label)
        click_layout.addRow("Flat index:", self.index_label)
        click_layout.addRow("Selected actuator:", self.selected_act_label)
        click_layout.addRow("Selected amplitude:", self.selected_amp_label)
        click_layout.addRow("Nonzero actuators:", self.n_active_label)
        right_layout.addWidget(click_box)

        contrib_box = QtWidgets.QGroupBox("Current actuator offsets")
        contrib_layout = QtWidgets.QVBoxLayout(contrib_box)
        self.contrib_text = QtWidgets.QPlainTextEdit()
        self.contrib_text.setReadOnly(True)
        self.contrib_text.setMinimumHeight(110)
        self.contrib_text.setMaximumHeight(140)
        contrib_layout.addWidget(self.contrib_text)
        right_layout.addWidget(contrib_box)

        dm_box = QtWidgets.QGroupBox("DM command preview (12x12)")
        dm_layout = QtWidgets.QVBoxLayout(dm_box)
        self.dm_view = pg.GraphicsLayoutWidget()
        self.dm_view.setMinimumHeight(180)
        self.dm_view.setMaximumHeight(230)
        self.dm_plot = self.dm_view.addViewBox(lockAspect=True, invertY=False)
        self.dm_img = pg.ImageItem(axisOrder="row-major")
        self.dm_plot.addItem(self.dm_img)
        dm_layout.addWidget(self.dm_view)
        right_layout.addWidget(dm_box, stretch=0)

        self.status_bar = QtWidgets.QLabel()
        right_layout.addWidget(self.status_bar)
        right_layout.addStretch(1)

        layout.addWidget(self.graphics, stretch=3)
        layout.addWidget(right, stretch=0)

    def _connect_signals(self) -> None:
        self.image_source.image_ready.connect(self._on_image)
        self.image_item.clicked.connect(self._on_image_clicked)
        self.scatter_all.sigClicked.connect(self._on_actuator_scatter_clicked)
        self.scatter_active.sigClicked.connect(self._on_actuator_scatter_clicked)
        self.scatter_selected.sigClicked.connect(self._on_actuator_scatter_clicked)

        self.amp_slider.valueChanged.connect(self._on_amp_slider)
        self.send_checkbox.toggled.connect(self._on_send_toggled)
        self.clear_selected_button.clicked.connect(self.controller.clear_selected_actuator)
        self.clear_all_button.clicked.connect(self.controller.clear_all_offsets)
        self.zero_button.clicked.connect(self.controller.zero_and_send)
        self.controller.state_changed.connect(self._on_state_changed)

    def closeEvent(self, event):  # type: ignore[override]
        try:
            self.image_source.stop()
        finally:
            self.controller.dm_sink.close()
        super().closeEvent(event)

    def _amp_from_slider(self, value: int) -> float:
        return value / 10000.0

    def _slider_from_amp(self, amp: float) -> int:
        return int(np.clip(np.round(amp * 10000.0), -1000, 1000))

    def _on_amp_slider(self, value: int) -> None:
        if self._ignore_slider_signal:
            return
        amp = self._amp_from_slider(int(value))
        self.amp_value_label.setText(f"{amp:+.4f}")
        self.controller.set_selected_amplitude(amp)

    def _sync_slider_to_selected_actuator(self) -> None:
        amp = self.controller.get_selected_amplitude()
        self._ignore_slider_signal = True
        self.amp_slider.setValue(self._slider_from_amp(amp))
        self.amp_value_label.setText(f"{amp:+.4f}")
        self._ignore_slider_signal = False

    def _on_send_toggled(self, checked: bool) -> None:
        self.controller.send_enabled = bool(checked and self.mode == "live" and self.allow_send)
        self._update_status()
        if self.controller.send_enabled:
            self.controller._refresh_preview()

    def _on_image(self, image: np.ndarray) -> None:
        image = np.asarray(image, dtype=float)
        self.image_item.setImage(image, autoLevels=True)

        if not hasattr(self, "_view_initialized"):
            self.view.setRange(
                xRange=(0, self.calib.image_shape[1]),
                yRange=(0, self.calib.image_shape[0]),
                padding=0.02,
            )
            self._view_initialized = True

        self._render_click_marker()

    def _on_image_clicked(self, x: float, y: float) -> None:
        ny, nx = self.calib.image_shape
        col = int(np.round(x))
        row = int(np.round(y))
        if not (0 <= row < ny and 0 <= col < nx):
            return
        self.controller.set_click(row=row, col=col)

    def _on_actuator_scatter_clicked(self, plot, points):
        if not points:
            return
        pt = points[0]
        idx = pt.data()
        if idx is None:
            return
        self.controller.set_selected_actuator(int(idx))

    def _on_state_changed(self) -> None:
        self._sync_slider_to_selected_actuator()
        self._render_click_marker()
        self._render_selected_actuator()
        self._render_active_actuators()
        self._update_diagnostics()
        self._update_dm_preview()
        self._update_status()

    def _render_actuator_overlay(self) -> None:
        xy = self.calib.actuator_xy_pix
        spots = [
            {"pos": (float(x), float(y)), "data": int(i)}
            for i, (x, y) in enumerate(xy)
        ]
        self.scatter_all.setData(spots)

    def _render_selected_actuator(self) -> None:
        idx = self.controller.state.selected_actuator
        if idx is None:
            self.scatter_selected.setData([])
            return
        x, y = self.calib.actuator_xy_pix[idx]
        self.scatter_selected.setData([{"pos": (float(x), float(y)), "size": 18, "data": int(idx)}])

    def _render_active_actuators(self) -> None:
        offsets = self.controller.state.actuator_offsets
        mask = np.abs(offsets) > 0
        if not np.any(mask):
            self.scatter_active.setData([])
            return

        idxs = np.where(mask)[0]
        xy = self.calib.actuator_xy_pix[mask]
        vals = np.abs(offsets[mask])
        vmax = np.max(vals) if np.max(vals) > 0 else 1.0
        sizes = 8 + 10 * vals / vmax
        spots = [
            {"pos": (float(x), float(y)), "size": float(s), "data": int(i)}
            for i, (x, y), s in zip(idxs, xy, sizes)
        ]
        self.scatter_active.setData(spots)

    def _render_click_marker(self) -> None:
        row = self.controller.state.click_row
        col = self.controller.state.click_col
        if row is None or col is None:
            self.click_marker.setData([])
            return
        self.click_marker.setData([{"pos": (float(col), float(row))}])

    def _update_diagnostics(self) -> None:
        row = self.controller.state.click_row
        col = self.controller.state.click_col
        act = self.controller.state.selected_actuator
        offsets = self.controller.state.actuator_offsets
        ny, nx = self.calib.image_shape

        if row is None or col is None:
            self.pixel_label.setText("-")
            self.index_label.setText("-")
        else:
            flat_idx = row * nx + col
            self.pixel_label.setText(f"({row}, {col})")
            self.index_label.setText(str(flat_idx))

        self.selected_act_label.setText("-" if act is None else str(act))
        self.selected_amp_label.setText(f"{self.controller.get_selected_amplitude():+.4f}")
        self.n_active_label.setText(str(int(np.sum(np.abs(offsets) > 0))))

        lines = []
        for act_idx, val in self.controller.top_contributors(8):
            lines.append(f"act {act_idx:3d}: {val:+.6e}")
        self.contrib_text.setPlainText("\n".join(lines))

    def _update_dm_preview(self) -> None:
        cmd = self.controller.state.preview_cmd
        if cmd is None:
            cmd = self.calib.flat_cmd
        grid = self._cmd_to_12x12(np.asarray(cmd, dtype=float).reshape(-1))
        self.dm_img.setImage(
            grid,
            autoLevels=False,
            levels=(-self.dm_display_limit, self.dm_display_limit),
        )

        if not hasattr(self, "_dm_view_initialized"):
            self.dm_plot.setRange(xRange=(0, 12), yRange=(0, 12), padding=0.04)
            self._dm_view_initialized = True

    def _update_status(self) -> None:
        if self.mode == "static":
            self.send_hint.setText("Static mode: hardware writes disabled.")
        elif self.mode == "live-readonly":
            self.send_hint.setText("Live-readonly mode: camera live, DM writes disabled.")
        elif self.mode == "live" and not self.allow_send:
            self.send_hint.setText("Live mode without --allow-send: DM writes disabled.")
        elif self.mode == "live" and self.allow_send and not self.controller.send_enabled:
            self.send_hint.setText("Live mode: enable checkbox to send commands.")
        else:
            self.send_hint.setText("Live mode: DM sending enabled.")

        send_txt = "ON" if self.controller.send_enabled else "OFF"
        self.status_bar.setText(
            f"Mode={self.mode} | Send={send_txt} | Active actuators={int(np.sum(np.abs(self.controller.state.actuator_offsets) > 0))}"
        )

    @staticmethod
    def _cmd_to_12x12(cmd: np.ndarray) -> np.ndarray:
        cmd = np.asarray(cmd, dtype=float).reshape(-1)
        if cmd.size == 140:
            out = np.full((12, 12), np.nan, dtype=float)
            keep = np.ones((12, 12), dtype=bool)
            keep[0, 0] = False
            keep[0, -1] = False
            keep[-1, 0] = False
            keep[-1, -1] = False
            out[keep] = cmd
            return out
        elif cmd.size == 144:
            return cmd.reshape(12, 12)
        else:
            raise ValueError(f"Expected command length 140 or 144, got {cmd.size}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive persistent DM click GUI using Baldr TOML calibration.")
    parser.add_argument("--toml", required=True, help="Path to beam TOML config file")
    parser.add_argument("--beam", type=int, required=True, help="Beam ID")
    parser.add_argument("--phasemask", default="H4", help="Phasemask key under beam{N} in TOML")

    parser.add_argument(
        "--mode",
        choices=["static", "live-readonly", "live"],
        default="live",
        help="GUI operating mode",
    )
    parser.add_argument(
        "--allow-send",
        action="store_true",
        default=True,
        help="Allow DM commands in live mode. Sending still requires enabling the GUI checkbox.",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    loader = TomlCalibrationLoader(toml_path=args.toml, beam_id=args.beam, phasemask=args.phasemask)
    calib = loader.load()

    if args.mode == "static":
        image_source = StaticImageSource(shape=calib.image_shape)
        dm_sink: DmSinkBase = NullDmSink()
    elif args.mode == "live-readonly":
        image_source = LiveShmImageSource(beam_id=calib.beam_id)
        dm_sink = NullDmSink()
    elif args.mode == "live":
        image_source = LiveShmImageSource(beam_id=calib.beam_id)
        dm_sink = LiveDmSink(beam_id=calib.beam_id) if args.allow_send else NullDmSink()
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major")

    controller = DmClickController(calib=calib, dm_sink=dm_sink)
    window = MainWindow(
        calib=calib,
        image_source=image_source,
        controller=controller,
        mode=args.mode,
        allow_send=bool(args.allow_send),
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

# #!/usr/bin/env python3
# """
# Interactive DM click GUI for Baldr/ZWFS calibration products.

# Behavior
# --------
# - Reads an existing Baldr TOML file.
# - Uses stored I2A only to infer actuator positions if explicit coordinates are absent.
# - Static mode uses a fixed random 32x32 image.
# - Clicking selects the nearest actuator in image pixel space.
# - The amplitude slider edits the currently selected actuator only.
# - Previously edited actuators retain their amplitudes by default.
# - Preview command = flat_cmd + actuator_offsets.
# - No recalibration is performed.
# """

# from __future__ import annotations

# import argparse
# import sys
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import pyqtgraph as pg
# import toml

# try:
#     from PyQt5 import QtCore, QtWidgets
# except Exception:
#     from PySide6 import QtCore, QtWidgets


# def _import_live_backends():
#     from xaosim.shmlib import shm  # type: ignore
#     from asgard_alignment.DM_shm_ctrl import dmclass  # type: ignore
#     return shm, dmclass


# @dataclass
# class CalibrationData:
#     beam_id: int
#     phasemask: str
#     image_shape: tuple[int, int]
#     i2a: np.ndarray
#     actuator_xy_pix: np.ndarray
#     flat_cmd: np.ndarray
#     reference_image: Optional[np.ndarray]
#     m2c: Optional[np.ndarray]


# class TomlCalibrationLoader:
#     def __init__(self, toml_path: str, beam_id: int, phasemask: str):
#         self.toml_path = Path(toml_path)
#         self.beam_id = int(beam_id)
#         self.beam_key = f"beam{self.beam_id}"
#         self.phasemask = phasemask

#     def load(self) -> CalibrationData:
#         if not self.toml_path.exists():
#             raise FileNotFoundError(f"TOML file not found: {self.toml_path}")

#         data = toml.load(self.toml_path)
#         if self.beam_key not in data:
#             raise KeyError(f"Missing TOML section '{self.beam_key}' in {self.toml_path}")

#         beam_data = data[self.beam_key]
#         i2a = self._load_i2a(beam_data)
#         ny, nx = self._infer_image_shape(i2a, beam_data)

#         if i2a.shape[1] != ny * nx:
#             raise ValueError(
#                 f"I2A has n_pix={i2a.shape[1]}, but inferred image shape is {(ny, nx)} -> {ny * nx} pixels"
#             )

#         reference_image = self._load_reference_image(beam_data, (ny, nx))
#         flat_cmd = self._load_flat_command(beam_data)
#         m2c = self._load_m2c(beam_data)
#         actuator_xy_pix = self._load_or_derive_actuator_positions(beam_data, i2a, (ny, nx))

#         n_act = i2a.shape[0]
#         if flat_cmd.size != n_act:
#             if flat_cmd.size == 140 and n_act != 140:
#                 raise ValueError(f"Flat command has length 140 but I2A has {n_act} actuators.")
#             if flat_cmd.size != 140 and n_act == 140:
#                 flat_cmd = np.zeros(140, dtype=float)
#             elif flat_cmd.size != n_act:
#                 flat_cmd = np.zeros(n_act, dtype=float)

#         return CalibrationData(
#             beam_id=self.beam_id,
#             phasemask=self.phasemask,
#             image_shape=(ny, nx),
#             i2a=i2a,
#             actuator_xy_pix=actuator_xy_pix,
#             flat_cmd=flat_cmd,
#             reference_image=reference_image,
#             m2c=m2c,
#         )

#     def _load_i2a(self, beam_data: dict) -> np.ndarray:
#         if "I2A" not in beam_data:
#             raise KeyError(f"Missing '{self.beam_key}.I2A' in TOML.")
#         i2a = np.asarray(beam_data["I2A"], dtype=float)
#         if i2a.ndim != 2:
#             raise ValueError(f"Expected I2A to be 2D, got shape {i2a.shape}")
#         return i2a

#     def _infer_image_shape(self, i2a: np.ndarray, beam_data: dict) -> tuple[int, int]:
#         n_pix = int(i2a.shape[1])
#         n_side = int(round(np.sqrt(n_pix)))

#         ctrl_model = (
#             beam_data.get(self.phasemask, {}).get("ctrl_model", {})
#             if isinstance(beam_data.get(self.phasemask, {}), dict)
#             else {}
#         )

#         for key in ("N0", "I0", "norm_pupil", "dark"):
#             if key in ctrl_model:
#                 arr = np.asarray(ctrl_model[key], dtype=float).reshape(-1)
#                 if arr.size == n_pix:
#                     return (n_side, n_side)

#         if n_side * n_side == n_pix:
#             return (n_side, n_side)

#         raise ValueError("Could not infer image shape from TOML.")

#     def _load_reference_image(
#         self,
#         beam_data: dict,
#         image_shape: tuple[int, int],
#     ) -> Optional[np.ndarray]:
#         ctrl_model = beam_data.get(self.phasemask, {}).get("ctrl_model", {})
#         for key in ("N0", "I0", "norm_pupil", "dark"):
#             if key in ctrl_model:
#                 arr = np.asarray(ctrl_model[key], dtype=float).reshape(-1)
#                 if arr.size == image_shape[0] * image_shape[1]:
#                     return arr.reshape(image_shape)
#         return None

#     def _load_flat_command(self, beam_data: dict) -> np.ndarray:
#         flat_cmd = np.zeros(140, dtype=float)

#         possible_paths = [
#             beam_data.get("dm_flat", None),
#             beam_data.get("DM_flat", None),
#             beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("dm_flat", None),
#             beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("DM_flat", None),
#         ]
#         for entry in possible_paths:
#             if entry is None:
#                 continue
#             if isinstance(entry, (str, bytes)):
#                 continue
#             try:
#                 arr = np.asarray(entry, dtype=float).reshape(-1)
#             except (TypeError, ValueError):
#                 continue
#             if arr.size == 140:
#                 flat_cmd = arr.copy()
#                 break
#             if arr.size == 144:
#                 flat_cmd = self._map_144_to_140(arr)
#                 break

#         return flat_cmd

#     def _load_m2c(self, beam_data: dict) -> Optional[np.ndarray]:
#         ctrl_model = beam_data.get(self.phasemask, {}).get("ctrl_model", {})
#         if "M2C" not in ctrl_model:
#             return None
#         m2c = np.asarray(ctrl_model["M2C"], dtype=float)
#         if m2c.ndim != 2:
#             return None
#         return m2c

#     def _load_or_derive_actuator_positions(
#         self,
#         beam_data: dict,
#         i2a: np.ndarray,
#         image_shape: tuple[int, int],
#     ) -> np.ndarray:
#         explicit_keys = [
#             "actuator_coord_list_pixel_space",
#             "actuator_coords_pixel_space",
#             "actuator_pixel_coords",
#         ]
#         for key in explicit_keys:
#             if key in beam_data:
#                 arr = np.asarray(beam_data[key], dtype=float)
#                 if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] == i2a.shape[0]:
#                     return arr

#         return self._derive_positions_from_i2a(i2a, image_shape)

#     @staticmethod
#     def _derive_positions_from_i2a(i2a: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
#         ny, nx = image_shape
#         yy, xx = np.indices((ny, nx), dtype=float)
#         xx_flat = xx.reshape(-1)
#         yy_flat = yy.reshape(-1)

#         coords = np.zeros((i2a.shape[0], 2), dtype=float)
#         for k, row in enumerate(i2a):
#             weights = np.abs(np.asarray(row, dtype=float).reshape(-1))
#             total = float(np.sum(weights))
#             if not np.isfinite(total) or total <= 0:
#                 peak = int(np.argmax(np.abs(row)))
#                 coords[k, 0] = xx_flat[peak]
#                 coords[k, 1] = yy_flat[peak]
#                 continue
#             coords[k, 0] = np.sum(weights * xx_flat) / total
#             coords[k, 1] = np.sum(weights * yy_flat) / total
#         return coords

#     @staticmethod
#     def _map_144_to_140(cmd144: np.ndarray) -> np.ndarray:
#         grid = np.asarray(cmd144, dtype=float).reshape(12, 12)
#         keep = np.ones((12, 12), dtype=bool)
#         keep[0, 0] = False
#         keep[0, -1] = False
#         keep[-1, 0] = False
#         keep[-1, -1] = False
#         return grid[keep].reshape(-1)


# class ImageSourceBase(QtCore.QObject):
#     image_ready = QtCore.pyqtSignal(object)

#     def start(self) -> None:
#         raise NotImplementedError

#     def stop(self) -> None:
#         raise NotImplementedError


# class StaticImageSource(ImageSourceBase):
#     def __init__(self, shape: tuple[int, int], seed: int = 1234):
#         super().__init__()
#         rng = np.random.default_rng(seed)
#         self._image = rng.normal(size=shape).astype(float)

#     def start(self) -> None:
#         self.image_ready.emit(self._image.copy())

#     def stop(self) -> None:
#         pass


# class LiveShmImageSource(ImageSourceBase):
#     def __init__(self, beam_id: int, timer_ms: int = 100):
#         super().__init__()
#         shm, _ = _import_live_backends()
#         self._shm = shm(f"/dev/shm/baldr{beam_id}.im.shm")
#         self._timer = QtCore.QTimer(self)
#         self._timer.setInterval(int(timer_ms))
#         self._timer.timeout.connect(self._poll)

#     def _poll(self) -> None:
#         try:
#             img = np.asarray(self._shm.get_data(), dtype=float)
#             self.image_ready.emit(img)
#         except Exception as exc:
#             print(f"Warning: failed to read SHM image: {exc}", file=sys.stderr)

#     def start(self) -> None:
#         self._poll()
#         self._timer.start()

#     def stop(self) -> None:
#         self._timer.stop()
#         try:
#             self._shm.close(erase_file=False)
#         except Exception:
#             pass


# class DmSinkBase:
#     def send(self, cmd: np.ndarray) -> None:
#         raise NotImplementedError

#     def close(self) -> None:
#         pass


# class NullDmSink(DmSinkBase):
#     def send(self, cmd: np.ndarray) -> None:
#         return


# class LiveDmSink(DmSinkBase):
#     def __init__(self, beam_id: int):
#         _, dmclass = _import_live_backends()
#         self._dm = dmclass(beam_id=beam_id)

#     def send(self, cmd: np.ndarray) -> None:
#         cmd = np.asarray(cmd, dtype=float).reshape(-1)
#         if cmd.size != 140:
#             raise ValueError(f"Expected 140-length DM command, got {cmd.size}")
#         cmd2d = self._dm.cmd_2_map2D(cmd, fill=0)
#         self._dm.set_data(cmd2d)


# @dataclass
# class ControllerState:
#     click_row: Optional[int] = None
#     click_col: Optional[int] = None
#     selected_actuator: Optional[int] = None
#     actuator_offsets: np.ndarray = field(default_factory=lambda: np.zeros(140, dtype=float))
#     preview_cmd: Optional[np.ndarray] = None


# class DmClickController(QtCore.QObject):
#     state_changed = QtCore.pyqtSignal()

#     def __init__(self, calib: CalibrationData, dm_sink: DmSinkBase):
#         super().__init__()
#         self.calib = calib
#         self.dm_sink = dm_sink
#         self.send_enabled = False

#         n_act = calib.i2a.shape[0]
#         self.state = ControllerState(actuator_offsets=np.zeros(n_act, dtype=float))
#         self._refresh_preview()

#     def set_click(self, row: int, col: int) -> None:
#         ny, nx = self.calib.image_shape
#         row = int(np.clip(row, 0, ny - 1))
#         col = int(np.clip(col, 0, nx - 1))

#         xy = self.calib.actuator_xy_pix
#         d2 = (xy[:, 0] - col) ** 2 + (xy[:, 1] - row) ** 2
#         nearest = int(np.argmin(d2))

#         self.state.click_row = row
#         self.state.click_col = col
#         self.state.selected_actuator = nearest
#         self.state_changed.emit()

#     def set_selected_amplitude(self, value: float) -> None:
#         if self.state.selected_actuator is None:
#             return
#         self.state.actuator_offsets[self.state.selected_actuator] = float(value)
#         self._refresh_preview()

#     def get_selected_amplitude(self) -> float:
#         if self.state.selected_actuator is None:
#             return 0.0
#         return float(self.state.actuator_offsets[self.state.selected_actuator])

#     def clear_selected_actuator(self) -> None:
#         if self.state.selected_actuator is None:
#             return
#         self.state.actuator_offsets[self.state.selected_actuator] = 0.0
#         self._refresh_preview()

#     def clear_all_offsets(self) -> None:
#         self.state.actuator_offsets[:] = 0.0
#         self._refresh_preview()

#     def zero_and_send(self) -> None:
#         self.state.actuator_offsets[:] = 0.0
#         self.state.preview_cmd = self.calib.flat_cmd.copy()
#         if self.send_enabled:
#             self.dm_sink.send(self.state.preview_cmd)
#         self.state_changed.emit()

#     def _refresh_preview(self) -> None:
#         self.state.preview_cmd = self.calib.flat_cmd + self.state.actuator_offsets
#         if self.send_enabled:
#             self.dm_sink.send(self.state.preview_cmd)
#         self.state_changed.emit()

#     def top_contributors(self, n: int = 10) -> list[tuple[int, float]]:
#         w = np.asarray(self.state.actuator_offsets, dtype=float).reshape(-1)
#         mask = np.abs(w) > 0
#         if not np.any(mask):
#             return []
#         order = np.argsort(np.abs(w))[::-1]
#         out = []
#         for i in order:
#             if np.abs(w[i]) <= 0:
#                 continue
#             out.append((int(i), float(w[i])))
#             if len(out) >= n:
#                 break
#         return out


# class ClickableImageItem(pg.ImageItem):
#     clicked = QtCore.pyqtSignal(float, float)

#     def mouseClickEvent(self, ev):  # type: ignore[override]
#         if ev.button() == QtCore.Qt.LeftButton:
#             pos = ev.pos()
#             self.clicked.emit(float(pos.x()), float(pos.y()))
#             ev.accept()
#         else:
#             super().mouseClickEvent(ev)


# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(
#         self,
#         calib: CalibrationData,
#         image_source: ImageSourceBase,
#         controller: DmClickController,
#         mode: str,
#         allow_send: bool,
#     ):
#         super().__init__()
#         self.calib = calib
#         self.image_source = image_source
#         self.controller = controller
#         self.mode = mode
#         self.allow_send = allow_send

#         self.dm_display_limit = 0.1
#         self._ignore_slider_signal = False

#         self.setWindowTitle(f"DM Click GUI | beam {calib.beam_id} | {mode}")
#         self.resize(1320, 860)

#         self._build_ui()
#         self._connect_signals()

#         self._render_actuator_overlay()
#         self._update_dm_preview()
#         self._update_status()

#         self.image_source.start()

#     def _build_ui(self) -> None:
#         central = QtWidgets.QWidget()
#         self.setCentralWidget(central)
#         layout = QtWidgets.QHBoxLayout(central)

#         self.graphics = pg.GraphicsLayoutWidget()
#         self.view = self.graphics.addViewBox(lockAspect=True, invertY=False)
#         self.view.setMouseEnabled(x=False, y=False)

#         self.image_item = ClickableImageItem(axisOrder="row-major")
#         self.view.addItem(self.image_item)

#         self.scatter_all = pg.ScatterPlotItem(size=10, pen=pg.mkPen("c", width=1.5), brush=None)
#         self.scatter_selected = pg.ScatterPlotItem(size=16, pen=pg.mkPen("y", width=2), brush=None)
#         self.scatter_active = pg.ScatterPlotItem(size=12, pen=pg.mkPen("r", width=2), brush=None)
#         self.click_marker = pg.ScatterPlotItem(size=12, pen=pg.mkPen("w", width=2), brush=None, symbol="x")

#         self.view.addItem(self.scatter_all)
#         self.view.addItem(self.scatter_active)
#         self.view.addItem(self.scatter_selected)
#         self.view.addItem(self.click_marker)

#         right = QtWidgets.QWidget()
#         right_layout = QtWidgets.QVBoxLayout(right)

#         mode_box = QtWidgets.QGroupBox("Mode")
#         mode_layout = QtWidgets.QFormLayout(mode_box)
#         self.mode_label = QtWidgets.QLabel(self.mode)
#         self.beam_label = QtWidgets.QLabel(str(self.calib.beam_id))
#         self.shape_label = QtWidgets.QLabel(f"{self.calib.image_shape[0]} x {self.calib.image_shape[1]}")
#         mode_layout.addRow("Mode:", self.mode_label)
#         mode_layout.addRow("Beam:", self.beam_label)
#         mode_layout.addRow("Image shape:", self.shape_label)
#         right_layout.addWidget(mode_box)

#         amp_box = QtWidgets.QGroupBox("Selected actuator amplitude")
#         amp_layout = QtWidgets.QVBoxLayout(amp_box)
#         self.amp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
#         self.amp_slider.setRange(-1000, 1000)
#         self.amp_slider.setValue(0)
#         self.amp_slider.setTickInterval(100)
#         self.amp_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
#         self.amp_value_label = QtWidgets.QLabel("0.0000")
#         amp_layout.addWidget(self.amp_slider)
#         amp_layout.addWidget(self.amp_value_label)
#         right_layout.addWidget(amp_box)

#         send_box = QtWidgets.QGroupBox("Commanding")
#         send_layout = QtWidgets.QVBoxLayout(send_box)
#         self.send_checkbox = QtWidgets.QCheckBox("Enable live send")
#         self.send_checkbox.setChecked(False)
#         self.send_checkbox.setEnabled(self.mode == "live" and self.allow_send)

#         self.send_hint = QtWidgets.QLabel()
#         self.clear_selected_button = QtWidgets.QPushButton("Clear selected actuator")
#         self.clear_all_button = QtWidgets.QPushButton("Clear all offsets")
#         self.zero_button = QtWidgets.QPushButton("Zero / send flat")

#         send_layout.addWidget(self.send_checkbox)
#         send_layout.addWidget(self.send_hint)
#         send_layout.addWidget(self.clear_selected_button)
#         send_layout.addWidget(self.clear_all_button)
#         send_layout.addWidget(self.zero_button)
#         right_layout.addWidget(send_box)

#         click_box = QtWidgets.QGroupBox("Selection diagnostics")
#         click_layout = QtWidgets.QFormLayout(click_box)
#         self.pixel_label = QtWidgets.QLabel("-")
#         self.index_label = QtWidgets.QLabel("-")
#         self.selected_act_label = QtWidgets.QLabel("-")
#         self.selected_amp_label = QtWidgets.QLabel("-")
#         self.n_active_label = QtWidgets.QLabel("0")
#         click_layout.addRow("Pixel (row, col):", self.pixel_label)
#         click_layout.addRow("Flat index:", self.index_label)
#         click_layout.addRow("Selected actuator:", self.selected_act_label)
#         click_layout.addRow("Selected amplitude:", self.selected_amp_label)
#         click_layout.addRow("Nonzero actuators:", self.n_active_label)
#         right_layout.addWidget(click_box)

#         contrib_box = QtWidgets.QGroupBox("Current actuator offsets")
#         contrib_layout = QtWidgets.QVBoxLayout(contrib_box)
#         self.contrib_text = QtWidgets.QPlainTextEdit()
#         self.contrib_text.setReadOnly(True)
#         self.contrib_text.setMinimumHeight(220)
#         contrib_layout.addWidget(self.contrib_text)
#         right_layout.addWidget(contrib_box)

#         dm_box = QtWidgets.QGroupBox("DM command preview (12x12)")
#         dm_layout = QtWidgets.QVBoxLayout(dm_box)
#         self.dm_view = pg.GraphicsLayoutWidget()
#         self.dm_plot = self.dm_view.addViewBox(lockAspect=True, invertY=False)
#         self.dm_img = pg.ImageItem(axisOrder="row-major")
#         self.dm_plot.addItem(self.dm_img)
#         dm_layout.addWidget(self.dm_view)
#         right_layout.addWidget(dm_box, stretch=1)

#         self.status_bar = QtWidgets.QLabel()
#         right_layout.addWidget(self.status_bar)
#         right_layout.addStretch(1)

#         layout.addWidget(self.graphics, stretch=3)
#         layout.addWidget(right, stretch=2)

#     def _connect_signals(self) -> None:
#         self.image_source.image_ready.connect(self._on_image)
#         self.image_item.clicked.connect(self._on_image_clicked)
#         self.amp_slider.valueChanged.connect(self._on_amp_slider)
#         self.send_checkbox.toggled.connect(self._on_send_toggled)
#         self.clear_selected_button.clicked.connect(self.controller.clear_selected_actuator)
#         self.clear_all_button.clicked.connect(self.controller.clear_all_offsets)
#         self.zero_button.clicked.connect(self.controller.zero_and_send)
#         self.controller.state_changed.connect(self._on_state_changed)

#     def closeEvent(self, event):  # type: ignore[override]
#         try:
#             self.image_source.stop()
#         finally:
#             self.controller.dm_sink.close()
#         super().closeEvent(event)

#     def _amp_from_slider(self, value: int) -> float:
#         return value / 10000.0

#     def _slider_from_amp(self, amp: float) -> int:
#         return int(np.clip(np.round(amp * 10000.0), -1000, 1000))

#     def _on_amp_slider(self, value: int) -> None:
#         if self._ignore_slider_signal:
#             return
#         amp = self._amp_from_slider(int(value))
#         self.amp_value_label.setText(f"{amp:+.4f}")
#         self.controller.set_selected_amplitude(amp)

#     def _sync_slider_to_selected_actuator(self) -> None:
#         amp = self.controller.get_selected_amplitude()
#         self._ignore_slider_signal = True
#         self.amp_slider.setValue(self._slider_from_amp(amp))
#         self.amp_value_label.setText(f"{amp:+.4f}")
#         self._ignore_slider_signal = False

#     def _on_send_toggled(self, checked: bool) -> None:
#         self.controller.send_enabled = bool(checked and self.mode == "live" and self.allow_send)
#         self._update_status()
#         if self.controller.send_enabled:
#             self.controller._refresh_preview()

#     def _on_image(self, image: np.ndarray) -> None:
#         image = np.asarray(image, dtype=float)
#         self.image_item.setImage(image, autoLevels=True)

#         if not hasattr(self, "_view_initialized"):
#             self.view.setRange(
#                 xRange=(0, self.calib.image_shape[1]),
#                 yRange=(0, self.calib.image_shape[0]),
#                 padding=0.02,
#             )
#             self._view_initialized = True

#         self._render_click_marker()

#     def _on_image_clicked(self, x: float, y: float) -> None:
#         ny, nx = self.calib.image_shape
#         col = int(np.round(x))
#         row = int(np.round(y))
#         if not (0 <= row < ny and 0 <= col < nx):
#             return
#         self.controller.set_click(row=row, col=col)

#     def _on_state_changed(self) -> None:
#         self._sync_slider_to_selected_actuator()
#         self._render_click_marker()
#         self._render_selected_actuator()
#         self._render_active_actuators()
#         self._update_diagnostics()
#         self._update_dm_preview()
#         self._update_status()

#     def _render_actuator_overlay(self) -> None:
#         xy = self.calib.actuator_xy_pix
#         spots = [{"pos": (float(x), float(y))} for x, y in xy]
#         self.scatter_all.setData(spots)

#     def _render_selected_actuator(self) -> None:
#         idx = self.controller.state.selected_actuator
#         if idx is None:
#             self.scatter_selected.setData([])
#             return
#         x, y = self.calib.actuator_xy_pix[idx]
#         self.scatter_selected.setData([{"pos": (float(x), float(y)), "size": 18}])

#     def _render_active_actuators(self) -> None:
#         offsets = self.controller.state.actuator_offsets
#         mask = np.abs(offsets) > 0
#         if not np.any(mask):
#             self.scatter_active.setData([])
#             return

#         xy = self.calib.actuator_xy_pix[mask]
#         vals = np.abs(offsets[mask])
#         vmax = np.max(vals) if np.max(vals) > 0 else 1.0
#         sizes = 8 + 10 * vals / vmax
#         spots = [
#             {"pos": (float(x), float(y)), "size": float(s)}
#             for (x, y), s in zip(xy, sizes)
#         ]
#         self.scatter_active.setData(spots)

#     def _render_click_marker(self) -> None:
#         row = self.controller.state.click_row
#         col = self.controller.state.click_col
#         if row is None or col is None:
#             self.click_marker.setData([])
#             return
#         self.click_marker.setData([{"pos": (float(col), float(row))}])

#     def _update_diagnostics(self) -> None:
#         row = self.controller.state.click_row
#         col = self.controller.state.click_col
#         act = self.controller.state.selected_actuator
#         offsets = self.controller.state.actuator_offsets
#         ny, nx = self.calib.image_shape

#         if row is None or col is None:
#             self.pixel_label.setText("-")
#             self.index_label.setText("-")
#         else:
#             flat_idx = row * nx + col
#             self.pixel_label.setText(f"({row}, {col})")
#             self.index_label.setText(str(flat_idx))

#         self.selected_act_label.setText("-" if act is None else str(act))
#         self.selected_amp_label.setText(f"{self.controller.get_selected_amplitude():+.4f}")
#         self.n_active_label.setText(str(int(np.sum(np.abs(offsets) > 0))))

#         lines = []
#         for act_idx, val in self.controller.top_contributors(12):
#             lines.append(f"act {act_idx:3d}: {val:+.6e}")
#         self.contrib_text.setPlainText("\n".join(lines))

#     def _update_dm_preview(self) -> None:
#         cmd = self.controller.state.preview_cmd
#         if cmd is None:
#             cmd = self.calib.flat_cmd
#         grid = self._cmd_to_12x12(np.asarray(cmd, dtype=float).reshape(-1))
#         self.dm_img.setImage(
#             grid,
#             autoLevels=False,
#             levels=(-self.dm_display_limit, self.dm_display_limit),
#         )

#         if not hasattr(self, "_dm_view_initialized"):
#             self.dm_plot.setRange(xRange=(0, 12), yRange=(0, 12), padding=0.04)
#             self._dm_view_initialized = True

#     def _update_status(self) -> None:
#         if self.mode == "static":
#             self.send_hint.setText("Static mode: hardware writes disabled.")
#         elif self.mode == "live-readonly":
#             self.send_hint.setText("Live-readonly mode: camera live, DM writes disabled.")
#         elif self.mode == "live" and not self.allow_send:
#             self.send_hint.setText("Live mode without --allow-send: DM writes disabled.")
#         elif self.mode == "live" and self.allow_send and not self.controller.send_enabled:
#             self.send_hint.setText("Live mode: enable checkbox to send commands.")
#         else:
#             self.send_hint.setText("Live mode: DM sending enabled.")

#         send_txt = "ON" if self.controller.send_enabled else "OFF"
#         self.status_bar.setText(
#             f"Mode={self.mode} | Send={send_txt} | Active actuators={int(np.sum(np.abs(self.controller.state.actuator_offsets) > 0))}"
#         )

#     @staticmethod
#     def _cmd_to_12x12(cmd: np.ndarray) -> np.ndarray:
#         cmd = np.asarray(cmd, dtype=float).reshape(-1)
#         if cmd.size == 140:
#             out = np.full((12, 12), np.nan, dtype=float)
#             keep = np.ones((12, 12), dtype=bool)
#             keep[0, 0] = False
#             keep[0, -1] = False
#             keep[-1, 0] = False
#             keep[-1, -1] = False
#             out[keep] = cmd
#             return out
#         elif cmd.size == 144:
#             return cmd.reshape(12, 12)
#         else:
#             raise ValueError(f"Expected command length 140 or 144, got {cmd.size}")


# def build_argparser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(description="Interactive persistent DM click GUI using Baldr TOML calibration.")
#     parser.add_argument("--toml", required=True, help="Path to beam TOML config file")
#     parser.add_argument("--beam", type=int, required=True, help="Beam ID")
#     parser.add_argument("--phasemask", default="H4", help="Phasemask key under beam{N} in TOML")
#     parser.add_argument(
#         "--mode",
#         choices=["static", "live-readonly", "live"],
#         default="static",
#         help="GUI operating mode",
#     )
#     parser.add_argument(
#         "--allow-send",
#         action="store_true",
#         help="Allow DM commands in live mode. Sending still requires enabling the GUI checkbox.",
#     )
#     return parser


# def main() -> int:
#     parser = build_argparser()
#     args = parser.parse_args()

#     loader = TomlCalibrationLoader(toml_path=args.toml, beam_id=args.beam, phasemask=args.phasemask)
#     calib = loader.load()

#     if args.mode == "static":
#         image_source = StaticImageSource(shape=calib.image_shape)
#         dm_sink: DmSinkBase = NullDmSink()
#     elif args.mode == "live-readonly":
#         image_source = LiveShmImageSource(beam_id=calib.beam_id)
#         dm_sink = NullDmSink()
#     elif args.mode == "live":
#         image_source = LiveShmImageSource(beam_id=calib.beam_id)
#         dm_sink = LiveDmSink(beam_id=calib.beam_id) if args.allow_send else NullDmSink()
#     else:
#         raise ValueError(f"Unsupported mode: {args.mode}")

#     app = QtWidgets.QApplication(sys.argv)
#     pg.setConfigOptions(imageAxisOrder="row-major")

#     controller = DmClickController(calib=calib, dm_sink=dm_sink)
#     window = MainWindow(
#         calib=calib,
#         image_source=image_source,
#         controller=controller,
#         mode=args.mode,
#         allow_send=bool(args.allow_send),
#     )
#     window.show()
#     return app.exec_()


# if __name__ == "__main__":
#     raise SystemExit(main())
# # #!/usr/bin/env python3
# # """
# # Interactive DM click GUI for Baldr/ZWFS calibration products.

# # This tool reads an existing Baldr TOML calibration file, loads the stored I2A
# # matrix, displays either a static test image or a live camera stream, overlays
# # DM actuator locations in pixel space, and lets the user click on the detector.

# # Updated behavior:
# # - Click selects the nearest actuator only.
# # - Static mode uses a fixed random 32x32 image.
# # - DM preview uses fixed display limits tied to slider range.
# # - No recalibration is performed.
# # """

# # from __future__ import annotations

# # import argparse
# # import sys
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Optional

# # import numpy as np
# # import pyqtgraph as pg
# # import toml

# # try:
# #     from PyQt5 import QtCore, QtWidgets
# # except Exception:
# #     from PySide6 import QtCore, QtWidgets


# # def _import_live_backends():
# #     from xaosim.shmlib import shm  # type: ignore
# #     from asgard_alignment.DM_shm_ctrl import dmclass  # type: ignore
# #     return shm, dmclass


# # @dataclass
# # class CalibrationData:
# #     beam_id: int
# #     phasemask: str
# #     image_shape: tuple[int, int]
# #     i2a: np.ndarray                  # (n_act, n_pix)
# #     actuator_xy_pix: np.ndarray      # (n_act, 2) in image pixel coordinates: x, y
# #     flat_cmd: np.ndarray             # (140,)
# #     reference_image: Optional[np.ndarray]
# #     m2c: Optional[np.ndarray]


# # class TomlCalibrationLoader:
# #     def __init__(self, toml_path: str, beam_id: int, phasemask: str):
# #         self.toml_path = Path(toml_path)
# #         self.beam_id = int(beam_id)
# #         self.beam_key = f"beam{self.beam_id}"
# #         self.phasemask = phasemask

# #     def load(self) -> CalibrationData:
# #         if not self.toml_path.exists():
# #             raise FileNotFoundError(f"TOML file not found: {self.toml_path}")

# #         data = toml.load(self.toml_path)
# #         if self.beam_key not in data:
# #             raise KeyError(f"Missing TOML section '{self.beam_key}' in {self.toml_path}")

# #         beam_data = data[self.beam_key]
# #         i2a = self._load_i2a(beam_data)
# #         ny, nx = self._infer_image_shape(i2a, beam_data)
# #         if i2a.shape[1] != ny * nx:
# #             raise ValueError(
# #                 f"I2A has n_pix={i2a.shape[1]}, but inferred image shape is {(ny, nx)} -> {ny*nx} pixels"
# #             )

# #         reference_image = self._load_reference_image(beam_data, (ny, nx))
# #         flat_cmd = self._load_flat_command(beam_data)
# #         m2c = self._load_m2c(beam_data)
# #         actuator_xy_pix = self._load_or_derive_actuator_positions(beam_data, i2a, (ny, nx))

# #         return CalibrationData(
# #             beam_id=self.beam_id,
# #             phasemask=self.phasemask,
# #             image_shape=(ny, nx),
# #             i2a=i2a,
# #             actuator_xy_pix=actuator_xy_pix,
# #             flat_cmd=flat_cmd,
# #             reference_image=reference_image,
# #             m2c=m2c,
# #         )

# #     def _load_i2a(self, beam_data: dict) -> np.ndarray:
# #         if "I2A" not in beam_data:
# #             raise KeyError(
# #                 f"Missing '{self.beam_key}.I2A' in TOML. This GUI requires a stored I2A matrix."
# #             )
# #         i2a = np.asarray(beam_data["I2A"], dtype=float)
# #         if i2a.ndim != 2:
# #             raise ValueError(f"Expected I2A to be 2D, got shape {i2a.shape}")
# #         return i2a

# #     def _infer_image_shape(self, i2a: np.ndarray, beam_data: dict) -> tuple[int, int]:
# #         n_pix = int(i2a.shape[1])
# #         n_side = int(round(np.sqrt(n_pix)))

# #         ctrl_model = (
# #             beam_data.get(self.phasemask, {}).get("ctrl_model", {})
# #             if isinstance(beam_data.get(self.phasemask, {}), dict)
# #             else {}
# #         )

# #         for key in ("N0", "I0", "norm_pupil", "dark"):
# #             if key in ctrl_model:
# #                 arr = np.asarray(ctrl_model[key], dtype=float).reshape(-1)
# #                 if arr.size == n_pix:
# #                     return (n_side, n_side)

# #         if n_side * n_side == n_pix:
# #             return (n_side, n_side)

# #         raise ValueError(
# #             "Could not infer image shape from TOML. Please ensure I2A corresponds to a square ROI."
# #         )

# #     def _load_reference_image(
# #         self,
# #         beam_data: dict,
# #         image_shape: tuple[int, int],
# #     ) -> Optional[np.ndarray]:
# #         ctrl_model = beam_data.get(self.phasemask, {}).get("ctrl_model", {})
# #         for key in ("N0", "I0", "norm_pupil", "dark"):
# #             if key in ctrl_model:
# #                 arr = np.asarray(ctrl_model[key], dtype=float).reshape(-1)
# #                 if arr.size == image_shape[0] * image_shape[1]:
# #                     return arr.reshape(image_shape)
# #         return None

# #     def _load_flat_command(self, beam_data: dict) -> np.ndarray:
# #         flat_cmd = np.zeros(140, dtype=float)

# #         possible_paths = [
# #             beam_data.get("dm_flat", None),
# #             beam_data.get("DM_flat", None),
# #             beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("dm_flat", None),
# #             beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("DM_flat", None),
# #         ]
# #         for entry in possible_paths:
# #             if entry is None:
# #                 continue
# #             if isinstance(entry, (str, bytes)):
# #                 continue
# #             try:
# #                 arr = np.asarray(entry, dtype=float).reshape(-1)
# #             except (TypeError, ValueError):
# #                 continue
# #             if arr.size == 140:
# #                 flat_cmd = arr.copy()
# #                 break
# #             if arr.size == 144:
# #                 flat_cmd = self._map_144_to_140(arr)
# #                 break

# #         return flat_cmd

# #     def _load_m2c(self, beam_data: dict) -> Optional[np.ndarray]:
# #         ctrl_model = beam_data.get(self.phasemask, {}).get("ctrl_model", {})
# #         if "M2C" not in ctrl_model:
# #             return None
# #         m2c = np.asarray(ctrl_model["M2C"], dtype=float)
# #         if m2c.ndim != 2:
# #             return None
# #         return m2c

# #     def _load_or_derive_actuator_positions(
# #         self,
# #         beam_data: dict,
# #         i2a: np.ndarray,
# #         image_shape: tuple[int, int],
# #     ) -> np.ndarray:
# #         explicit_keys = [
# #             "actuator_coord_list_pixel_space",
# #             "actuator_coords_pixel_space",
# #             "actuator_pixel_coords",
# #         ]
# #         for key in explicit_keys:
# #             if key in beam_data:
# #                 arr = np.asarray(beam_data[key], dtype=float)
# #                 if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] == i2a.shape[0]:
# #                     return arr

# #         return self._derive_positions_from_i2a(i2a, image_shape)

# #     @staticmethod
# #     def _derive_positions_from_i2a(i2a: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
# #         ny, nx = image_shape
# #         yy, xx = np.indices((ny, nx), dtype=float)
# #         xx_flat = xx.reshape(-1)
# #         yy_flat = yy.reshape(-1)

# #         coords = np.zeros((i2a.shape[0], 2), dtype=float)
# #         for k, row in enumerate(i2a):
# #             weights = np.abs(np.asarray(row, dtype=float).reshape(-1))
# #             total = float(np.sum(weights))
# #             if not np.isfinite(total) or total <= 0:
# #                 peak = int(np.argmax(np.abs(row)))
# #                 coords[k, 0] = xx_flat[peak]
# #                 coords[k, 1] = yy_flat[peak]
# #                 continue
# #             coords[k, 0] = np.sum(weights * xx_flat) / total
# #             coords[k, 1] = np.sum(weights * yy_flat) / total
# #         return coords

# #     @staticmethod
# #     def _map_144_to_140(cmd144: np.ndarray) -> np.ndarray:
# #         grid = np.asarray(cmd144, dtype=float).reshape(12, 12)
# #         keep = np.ones((12, 12), dtype=bool)
# #         keep[0, 0] = False
# #         keep[0, -1] = False
# #         keep[-1, 0] = False
# #         keep[-1, -1] = False
# #         return grid[keep].reshape(-1)


# # class ImageSourceBase(QtCore.QObject):
# #     image_ready = QtCore.pyqtSignal(object)

# #     def start(self) -> None:
# #         raise NotImplementedError

# #     def stop(self) -> None:
# #         raise NotImplementedError

# #     def latest(self) -> np.ndarray:
# #         raise NotImplementedError


# # class StaticImageSource(ImageSourceBase):
# #     def __init__(self, shape: tuple[int, int], seed: int = 1234):
# #         super().__init__()
# #         rng = np.random.default_rng(seed)
# #         self._image = rng.normal(size=shape).astype(float)

# #     def start(self) -> None:
# #         self.image_ready.emit(self._image.copy())

# #     def stop(self) -> None:
# #         pass

# #     def latest(self) -> np.ndarray:
# #         return self._image.copy()


# # class LiveShmImageSource(ImageSourceBase):
# #     def __init__(self, beam_id: int, timer_ms: int = 100):
# #         super().__init__()
# #         shm, _ = _import_live_backends()
# #         self._shm = shm(f"/dev/shm/baldr{beam_id}.im.shm")
# #         self._timer = QtCore.QTimer(self)
# #         self._timer.setInterval(int(timer_ms))
# #         self._timer.timeout.connect(self._poll)
# #         self._latest: Optional[np.ndarray] = None

# #     def _poll(self) -> None:
# #         try:
# #             img = np.asarray(self._shm.get_data(), dtype=float)
# #             self._latest = img
# #             self.image_ready.emit(img)
# #         except Exception as exc:
# #             print(f"Warning: failed to read SHM image: {exc}", file=sys.stderr)

# #     def start(self) -> None:
# #         self._poll()
# #         self._timer.start()

# #     def stop(self) -> None:
# #         self._timer.stop()
# #         try:
# #             self._shm.close(erase_file=False)
# #         except Exception:
# #             pass

# #     def latest(self) -> np.ndarray:
# #         if self._latest is None:
# #             raise RuntimeError("No live image has been received yet.")
# #         return self._latest.copy()


# # class DmSinkBase:
# #     def send(self, cmd140: np.ndarray) -> None:
# #         raise NotImplementedError

# #     def close(self) -> None:
# #         pass


# # class NullDmSink(DmSinkBase):
# #     def send(self, cmd140: np.ndarray) -> None:
# #         return


# # class LiveDmSink(DmSinkBase):
# #     def __init__(self, beam_id: int):
# #         _, dmclass = _import_live_backends()
# #         self._dm = dmclass(beam_id=beam_id)

# #     def send(self, cmd140: np.ndarray) -> None:
# #         cmd140 = np.asarray(cmd140, dtype=float).reshape(-1)
# #         if cmd140.size != 140:
# #             raise ValueError(f"Expected 140-length DM command, got {cmd140.size}")
# #         cmd2d = self._dm.cmd_2_map2D(cmd140, fill=0)
# #         self._dm.set_data(cmd2d)


# # @dataclass
# # class ControllerState:
# #     click_row: Optional[int] = None
# #     click_col: Optional[int] = None
# #     nearest_actuator: Optional[int] = None
# #     weights: Optional[np.ndarray] = None
# #     amplitude: float = 0.0
# #     preview_cmd: Optional[np.ndarray] = None


# # class DmClickController(QtCore.QObject):
# #     state_changed = QtCore.pyqtSignal()

# #     def __init__(self, calib: CalibrationData, dm_sink: DmSinkBase):
# #         super().__init__()
# #         self.calib = calib
# #         self.dm_sink = dm_sink
# #         self.state = ControllerState()
# #         self.send_enabled = False

# #     def set_amplitude(self, value: float) -> None:
# #         self.state.amplitude = float(value)
# #         self._refresh_preview()

# #     def set_click(self, row: int, col: int) -> None:
# #         ny, nx = self.calib.image_shape
# #         row = int(np.clip(row, 0, ny - 1))
# #         col = int(np.clip(col, 0, nx - 1))

# #         # Nearest-actuator selection in pixel space
# #         xy = self.calib.actuator_xy_pix
# #         d2 = (xy[:, 0] - col) ** 2 + (xy[:, 1] - row) ** 2
# #         nearest = int(np.argmin(d2))

# #         weights = np.zeros(self.calib.i2a.shape[0], dtype=float)
# #         weights[nearest] = 1.0

# #         self.state.click_row = row
# #         self.state.click_col = col
# #         self.state.nearest_actuator = nearest
# #         self.state.weights = weights
# #         self._refresh_preview()

# #     def clear(self) -> None:
# #         self.state = ControllerState(amplitude=self.state.amplitude)
# #         self.state_changed.emit()

# #     def zero_and_send(self) -> None:
# #         self.state.preview_cmd = self.calib.flat_cmd.copy()
# #         if self.send_enabled:
# #             self.dm_sink.send(self.state.preview_cmd)
# #         self.state_changed.emit()

# #     def _refresh_preview(self) -> None:
# #         if self.state.weights is None:
# #             self.state.preview_cmd = self.calib.flat_cmd.copy()
# #             self.state_changed.emit()
# #             return

# #         delta = self.state.amplitude * self.state.weights
# #         self.state.preview_cmd = self.calib.flat_cmd + delta

# #         if self.send_enabled:
# #             self.dm_sink.send(self.state.preview_cmd)

# #         self.state_changed.emit()

# #     def top_contributors(self, n: int = 8) -> list[tuple[int, float]]:
# #         if self.state.weights is None:
# #             return []
# #         w = np.asarray(self.state.weights, dtype=float).reshape(-1)
# #         order = np.argsort(np.abs(w))[::-1][:n]
# #         return [(int(i), float(w[i])) for i in order if np.abs(w[i]) > 0]


# # class ClickableImageItem(pg.ImageItem):
# #     clicked = QtCore.pyqtSignal(float, float)

# #     def mouseClickEvent(self, ev):  # type: ignore[override]
# #         if ev.button() == QtCore.Qt.LeftButton:
# #             pos = ev.pos()
# #             self.clicked.emit(float(pos.x()), float(pos.y()))
# #             ev.accept()
# #         else:
# #             super().mouseClickEvent(ev)


# # class MainWindow(QtWidgets.QMainWindow):
# #     def __init__(
# #         self,
# #         calib: CalibrationData,
# #         image_source: ImageSourceBase,
# #         controller: DmClickController,
# #         mode: str,
# #         allow_send: bool,
# #     ):
# #         super().__init__()
# #         self.calib = calib
# #         self.image_source = image_source
# #         self.controller = controller
# #         self.mode = mode
# #         self.allow_send = allow_send
# #         self.current_image: Optional[np.ndarray] = None

# #         self.dm_display_limit = 0.1  # matches slider mapping below

# #         self.setWindowTitle(f"DM Click GUI | beam {calib.beam_id} | {mode}")
# #         self.resize(1300, 820)
# #         self._build_ui()
# #         self._connect_signals()

# #         self._render_actuator_overlay()
# #         self._update_dm_preview()
# #         self._update_status()

# #         self.image_source.start()

# #     def _build_ui(self) -> None:
# #         central = QtWidgets.QWidget()
# #         self.setCentralWidget(central)
# #         layout = QtWidgets.QHBoxLayout(central)

# #         self.graphics = pg.GraphicsLayoutWidget()
# #         self.view = self.graphics.addViewBox(lockAspect=True, invertY=False)
# #         self.view.setMouseEnabled(x=False, y=False)

# #         self.image_item = ClickableImageItem(axisOrder="row-major")
# #         self.view.addItem(self.image_item)

# #         self.scatter_all = pg.ScatterPlotItem(size=10, pen=pg.mkPen("c", width=1.5), brush=None)
# #         self.scatter_hot = pg.ScatterPlotItem(size=16, pen=pg.mkPen("r", width=2), brush=None)
# #         self.click_marker = pg.ScatterPlotItem(size=12, pen=pg.mkPen("y", width=2), brush=None, symbol="x")

# #         self.view.addItem(self.scatter_all)
# #         self.view.addItem(self.scatter_hot)
# #         self.view.addItem(self.click_marker)

# #         right = QtWidgets.QWidget()
# #         right_layout = QtWidgets.QVBoxLayout(right)

# #         mode_box = QtWidgets.QGroupBox("Mode")
# #         mode_layout = QtWidgets.QFormLayout(mode_box)
# #         self.mode_label = QtWidgets.QLabel(self.mode)
# #         self.beam_label = QtWidgets.QLabel(str(self.calib.beam_id))
# #         self.shape_label = QtWidgets.QLabel(f"{self.calib.image_shape[0]} x {self.calib.image_shape[1]}")
# #         mode_layout.addRow("Mode:", self.mode_label)
# #         mode_layout.addRow("Beam:", self.beam_label)
# #         mode_layout.addRow("Image shape:", self.shape_label)
# #         right_layout.addWidget(mode_box)

# #         amp_box = QtWidgets.QGroupBox("Amplitude")
# #         amp_layout = QtWidgets.QVBoxLayout(amp_box)
# #         self.amp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
# #         self.amp_slider.setRange(-1000, 1000)
# #         self.amp_slider.setValue(0)
# #         self.amp_slider.setTickInterval(100)
# #         self.amp_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
# #         self.amp_value_label = QtWidgets.QLabel("0.0000")
# #         amp_layout.addWidget(self.amp_slider)
# #         amp_layout.addWidget(self.amp_value_label)
# #         right_layout.addWidget(amp_box)

# #         send_box = QtWidgets.QGroupBox("Commanding")
# #         send_layout = QtWidgets.QVBoxLayout(send_box)
# #         self.send_checkbox = QtWidgets.QCheckBox("Enable live send")
# #         self.send_checkbox.setChecked(False)
# #         self.send_checkbox.setEnabled(self.mode == "live" and self.allow_send)
# #         self.send_hint = QtWidgets.QLabel()
# #         self.zero_button = QtWidgets.QPushButton("Zero / send flat")
# #         self.clear_button = QtWidgets.QPushButton("Clear click")
# #         send_layout.addWidget(self.send_checkbox)
# #         send_layout.addWidget(self.send_hint)
# #         send_layout.addWidget(self.zero_button)
# #         send_layout.addWidget(self.clear_button)
# #         right_layout.addWidget(send_box)

# #         click_box = QtWidgets.QGroupBox("Click diagnostics")
# #         click_layout = QtWidgets.QFormLayout(click_box)
# #         self.pixel_label = QtWidgets.QLabel("-")
# #         self.index_label = QtWidgets.QLabel("-")
# #         self.nearest_act_label = QtWidgets.QLabel("-")
# #         self.norm_label = QtWidgets.QLabel("-")
# #         self.maxw_label = QtWidgets.QLabel("-")
# #         click_layout.addRow("Pixel (row, col):", self.pixel_label)
# #         click_layout.addRow("Flat index:", self.index_label)
# #         click_layout.addRow("Nearest actuator:", self.nearest_act_label)
# #         click_layout.addRow("||weights||:", self.norm_label)
# #         click_layout.addRow("max |weight|:", self.maxw_label)
# #         right_layout.addWidget(click_box)

# #         contrib_box = QtWidgets.QGroupBox("Top actuator contributors")
# #         contrib_layout = QtWidgets.QVBoxLayout(contrib_box)
# #         self.contrib_text = QtWidgets.QPlainTextEdit()
# #         self.contrib_text.setReadOnly(True)
# #         self.contrib_text.setMinimumHeight(180)
# #         contrib_layout.addWidget(self.contrib_text)
# #         right_layout.addWidget(contrib_box)

# #         dm_box = QtWidgets.QGroupBox("DM command preview (12x12)")
# #         dm_layout = QtWidgets.QVBoxLayout(dm_box)
# #         self.dm_view = pg.GraphicsLayoutWidget()
# #         self.dm_plot = self.dm_view.addViewBox(lockAspect=True, invertY=False)
# #         self.dm_img = pg.ImageItem(axisOrder="row-major")
# #         self.dm_plot.addItem(self.dm_img)
# #         dm_layout.addWidget(self.dm_view)
# #         right_layout.addWidget(dm_box, stretch=1)

# #         self.status_bar = QtWidgets.QLabel()
# #         right_layout.addWidget(self.status_bar)
# #         right_layout.addStretch(1)

# #         layout.addWidget(self.graphics, stretch=3)
# #         layout.addWidget(right, stretch=2)

# #     def _connect_signals(self) -> None:
# #         self.image_source.image_ready.connect(self._on_image)
# #         self.image_item.clicked.connect(self._on_image_clicked)
# #         self.amp_slider.valueChanged.connect(self._on_amp_slider)
# #         self.send_checkbox.toggled.connect(self._on_send_toggled)
# #         self.zero_button.clicked.connect(self.controller.zero_and_send)
# #         self.clear_button.clicked.connect(self.controller.clear)
# #         self.controller.state_changed.connect(self._on_state_changed)

# #     def closeEvent(self, event):  # type: ignore[override]
# #         try:
# #             self.image_source.stop()
# #         finally:
# #             self.controller.dm_sink.close()
# #         super().closeEvent(event)

# #     def _amp_from_slider(self, value: int) -> float:
# #         return value / 10000.0  # slider ±1000 -> amplitude ±0.1

# #     def _on_amp_slider(self, value: int) -> None:
# #         amp = self._amp_from_slider(int(value))
# #         self.amp_value_label.setText(f"{amp:+.4f}")
# #         self.controller.set_amplitude(amp)

# #     def _on_send_toggled(self, checked: bool) -> None:
# #         self.controller.send_enabled = bool(checked and self.mode == "live" and self.allow_send)
# #         self._update_status()
# #         if self.controller.send_enabled:
# #             self.controller.set_amplitude(self.controller.state.amplitude)

# #     def _on_image(self, image: np.ndarray) -> None:
# #         image = np.asarray(image, dtype=float)
# #         self.current_image = image
# #         self.image_item.setImage(image, autoLevels=True)

# #         # Only set the view range once, on first image
# #         if not hasattr(self, "_view_initialized"):
# #             self.view.setRange(
# #                 xRange=(0, self.calib.image_shape[1]),
# #                 yRange=(0, self.calib.image_shape[0]),
# #                 padding=0.02,
# #             )
# #             self._view_initialized = True

# #         self._render_click_marker()

# #     def _on_image_clicked(self, x: float, y: float) -> None:
# #         ny, nx = self.calib.image_shape
# #         col = int(np.round(x))
# #         row = int(np.round(y))
# #         if not (0 <= row < ny and 0 <= col < nx):
# #             return
# #         self.controller.set_click(row=row, col=col)

# #     def _on_state_changed(self) -> None:
# #         self._render_click_marker()
# #         self._render_hot_actuators()
# #         self._update_click_diagnostics()
# #         self._update_dm_preview()
# #         self._update_status()

# #     def _render_actuator_overlay(self) -> None:
# #         xy = self.calib.actuator_xy_pix
# #         spots = [{"pos": (float(x), float(y))} for x, y in xy]
# #         self.scatter_all.setData(spots)

# #     def _render_hot_actuators(self) -> None:
# #         nearest = self.controller.state.nearest_actuator
# #         if nearest is None:
# #             self.scatter_hot.setData([])
# #             return
# #         x, y = self.calib.actuator_xy_pix[nearest]
# #         self.scatter_hot.setData([{"pos": (float(x), float(y)), "size": 16}])

# #     def _render_click_marker(self) -> None:
# #         row = self.controller.state.click_row
# #         col = self.controller.state.click_col
# #         if row is None or col is None:
# #             self.click_marker.setData([])
# #             return
# #         self.click_marker.setData([{"pos": (float(col), float(row))}])

# #     def _update_click_diagnostics(self) -> None:
# #         row = self.controller.state.click_row
# #         col = self.controller.state.click_col
# #         weights = self.controller.state.weights
# #         nearest = self.controller.state.nearest_actuator
# #         ny, nx = self.calib.image_shape

# #         if row is None or col is None or weights is None:
# #             self.pixel_label.setText("-")
# #             self.index_label.setText("-")
# #             self.nearest_act_label.setText("-")
# #             self.norm_label.setText("-")
# #             self.maxw_label.setText("-")
# #             self.contrib_text.setPlainText("")
# #             return

# #         flat_idx = row * nx + col
# #         self.pixel_label.setText(f"({row}, {col})")
# #         self.index_label.setText(str(flat_idx))
# #         self.nearest_act_label.setText(str(nearest) if nearest is not None else "-")
# #         self.norm_label.setText(f"{np.linalg.norm(weights):.4e}")
# #         self.maxw_label.setText(f"{np.max(np.abs(weights)):.4e}")

# #         lines = []
# #         for act, val in self.controller.top_contributors(n=10):
# #             lines.append(f"act {act:3d}: {val:+.6e}")
# #         self.contrib_text.setPlainText("\n".join(lines))

# #     def _update_dm_preview(self) -> None:
# #         cmd = self.controller.state.preview_cmd
# #         if cmd is None:
# #             cmd = self.calib.flat_cmd
# #         grid = self._cmd140_to_12x12(np.asarray(cmd, dtype=float).reshape(-1))
# #         self.dm_img.setImage(
# #             grid,
# #             autoLevels=False,
# #             levels=(-self.dm_display_limit, self.dm_display_limit),
# #         )

# #         if not hasattr(self, "_dm_view_initialized"):
# #             self.dm_plot.setRange(xRange=(0, 12), yRange=(0, 12), padding=0.04)
# #             self._dm_view_initialized = True

# #     def _update_status(self) -> None:
# #         if self.mode == "static":
# #             self.send_hint.setText("Static mode: hardware writes disabled.")
# #         elif self.mode == "live-readonly":
# #             self.send_hint.setText("Live-readonly mode: camera live, DM writes disabled.")
# #         elif self.mode == "live" and not self.allow_send:
# #             self.send_hint.setText("Live mode without --allow-send: DM writes disabled.")
# #         elif self.mode == "live" and self.allow_send and not self.controller.send_enabled:
# #             self.send_hint.setText("Live mode: enable checkbox to send commands.")
# #         else:
# #             self.send_hint.setText("Live mode: DM sending enabled.")

# #         mode_txt = self.mode
# #         send_txt = "ON" if self.controller.send_enabled else "OFF"
# #         self.status_bar.setText(
# #             f"Mode={mode_txt} | Send={send_txt} | Amplitude={self.controller.state.amplitude:+.4f}"
# #         )

# #     @staticmethod
# #     def _cmd140_to_12x12(cmd140: np.ndarray) -> np.ndarray:
# #         cmd140 = np.asarray(cmd140, dtype=float).reshape(-1)
# #         if cmd140.size != 140:
# #             raise ValueError(f"Expected 140-length command, got {cmd140.size}")
# #         out = np.full((12, 12), np.nan, dtype=float)
# #         keep = np.ones((12, 12), dtype=bool)
# #         keep[0, 0] = False
# #         keep[0, -1] = False
# #         keep[-1, 0] = False
# #         keep[-1, -1] = False
# #         out[keep] = cmd140
# #         return out


# # def build_argparser() -> argparse.ArgumentParser:
# #     parser = argparse.ArgumentParser(description="Interactive nearest-actuator DM GUI using Baldr TOML calibration.")
# #     parser.add_argument("--toml", required=True, help="Path to beam TOML config file")
# #     parser.add_argument("--beam", type=int, required=True, help="Beam ID")
# #     parser.add_argument("--phasemask", default="H4", help="Phasemask key under beam{N} in TOML")
# #     parser.add_argument(
# #         "--mode",
# #         choices=["static", "live-readonly", "live"],
# #         default="static",
# #         help="GUI operating mode",
# #     )
# #     parser.add_argument(
# #         "--allow-send",
# #         action="store_true",
# #         help="Allow DM commands in live mode. Sending still requires enabling the GUI checkbox.",
# #     )
# #     return parser


# # def main() -> int:
# #     parser = build_argparser()
# #     args = parser.parse_args()

# #     loader = TomlCalibrationLoader(toml_path=args.toml, beam_id=args.beam, phasemask=args.phasemask)
# #     calib = loader.load()

# #     if args.mode == "static":
# #         image_source = StaticImageSource(shape=calib.image_shape)
# #         dm_sink: DmSinkBase = NullDmSink()
# #     elif args.mode == "live-readonly":
# #         image_source = LiveShmImageSource(beam_id=calib.beam_id)
# #         dm_sink = NullDmSink()
# #     elif args.mode == "live":
# #         image_source = LiveShmImageSource(beam_id=calib.beam_id)
# #         dm_sink = LiveDmSink(beam_id=calib.beam_id) if args.allow_send else NullDmSink()
# #     else:
# #         raise ValueError(f"Unsupported mode: {args.mode}")

# #     app = QtWidgets.QApplication(sys.argv)
# #     pg.setConfigOptions(imageAxisOrder="row-major")

# #     controller = DmClickController(calib=calib, dm_sink=dm_sink)
# #     window = MainWindow(
# #         calib=calib,
# #         image_source=image_source,
# #         controller=controller,
# #         mode=args.mode,
# #         allow_send=bool(args.allow_send),
# #     )
# #     window.show()
# #     return app.exec_()


# # if __name__ == "__main__":
# #     raise SystemExit(main())
# # # #!/usr/bin/env python3
# # # """
# # # Interactive DM click GUI for Baldr/ZWFS calibration products.

# # # Purpose
# # # -------
# # # This tool reads an existing Baldr TOML calibration file, loads the stored I2A
# # # (intensity-to-actuator) matrix, displays either a static test image or a live
# # # camera stream, overlays DM actuator locations in pixel space, and lets the user
# # # click on a detector pixel to generate a DM command from the corresponding I2A
# # # column.

# # # Key design choices
# # # ------------------
# # # - No recalibration is performed.
# # # - Click-to-actuator mapping uses the stored I2A directly.
# # # - Static mode uses a fixed random 32x32 image.
# # # - Live-readonly mode streams from SHM but never sends commands.
# # # - Live mode can optionally send commands to the DM.
# # # - Actuator overlay positions are derived from the I2A rows unless explicitly
# # #   provided in the TOML.

# # # Assumptions about TOML layout
# # # -----------------------------
# # # This script is written to work with the TOML structure produced by the user's
# # # Baldr calibration pipeline, where:

# # # - beam{beam_id}.I2A stores the bilinear interpolation matrix as nested lists.
# # # - beam{beam_id}.{phasemask}.ctrl_model.N0 or I0 may store a 32x32 flattened
# # #   reference image.
# # # - beam{beam_id}.{phasemask}.ctrl_model.M2C may optionally be present.

# # # If explicit actuator pixel coordinates are not present in the TOML, this script
# # # computes them from I2A by estimating a weighted pixel centroid per actuator.

# # # Dependencies
# # # ------------
# # # - numpy
# # # - toml
# # # - pyqtgraph
# # # - PyQt5 or PySide6

# # # Optional live-mode dependencies
# # # -------------------------------
# # # - xaosim.shmlib.shm
# # # - asgard_alignment.DM_shm_ctrl.dmclass

# # # Example usage
# # # -------------
# # # Static mode:
# # #     python dm_click_gui.py --toml /usr/local/etc/baldr/baldr_config_2.toml --beam 2 --mode static

# # # Live readonly:
# # #     python dm_click_gui.py --toml /usr/local/etc/baldr/baldr_config_2.toml --beam 2 --mode live-readonly

# # # Live control:
# # #     python dm_click_gui.py --toml /usr/local/etc/baldr/baldr_config_2.toml --beam 2 --mode live --allow-send
# # # """

# # # from __future__ import annotations

# # # import argparse
# # # import sys
# # # from dataclasses import dataclass
# # # from pathlib import Path
# # # from typing import Optional

# # # import numpy as np
# # # import pyqtgraph as pg
# # # import toml

# # # try:
# # #     from PyQt5 import QtCore, QtWidgets
# # # except Exception:  # pragma: no cover
# # #     from PySide6 import QtCore, QtWidgets


# # # def _import_live_backends():
# # #     """Import live-mode backends lazily so static mode has minimal requirements."""
# # #     from xaosim.shmlib import shm  # type: ignore
# # #     from asgard_alignment.DM_shm_ctrl import dmclass  # type: ignore
# # #     return shm, dmclass


# # # @dataclass
# # # class CalibrationData:
# # #     beam_id: int
# # #     phasemask: str
# # #     image_shape: tuple[int, int]
# # #     i2a: np.ndarray  # (n_act, n_pix)
# # #     actuator_xy_pix: np.ndarray  # (n_act, 2), x/y in image pixel space
# # #     flat_cmd: np.ndarray  # (140,)
# # #     reference_image: Optional[np.ndarray]
# # #     m2c: Optional[np.ndarray]


# # # class TomlCalibrationLoader:
# # #     def __init__(self, toml_path: str, beam_id: int, phasemask: str):
# # #         self.toml_path = Path(toml_path)
# # #         self.beam_id = int(beam_id)
# # #         self.beam_key = f"beam{self.beam_id}"
# # #         self.phasemask = phasemask

# # #     def load(self) -> CalibrationData:
# # #         if not self.toml_path.exists():
# # #             raise FileNotFoundError(f"TOML file not found: {self.toml_path}")

# # #         data = toml.load(self.toml_path)
# # #         if self.beam_key not in data:
# # #             raise KeyError(f"Missing TOML section '{self.beam_key}' in {self.toml_path}")

# # #         beam_data = data[self.beam_key]
# # #         i2a = self._load_i2a(beam_data)
# # #         ny, nx = self._infer_image_shape(i2a, beam_data, data)
# # #         if i2a.shape[1] != ny * nx:
# # #             raise ValueError(
# # #                 f"I2A has n_pix={i2a.shape[1]}, but inferred image shape is {(ny, nx)} -> {ny*nx} pixels"
# # #             )

# # #         reference_image = self._load_reference_image(data, beam_data, (ny, nx))
# # #         flat_cmd = self._load_flat_command(data, beam_data)
# # #         m2c = self._load_m2c(data, beam_data)
# # #         actuator_xy_pix = self._load_or_derive_actuator_positions(beam_data, i2a, (ny, nx))

# # #         return CalibrationData(
# # #             beam_id=self.beam_id,
# # #             phasemask=self.phasemask,
# # #             image_shape=(ny, nx),
# # #             i2a=i2a,
# # #             actuator_xy_pix=actuator_xy_pix,
# # #             flat_cmd=flat_cmd,
# # #             reference_image=reference_image,
# # #             m2c=m2c,
# # #         )

# # #     def _load_i2a(self, beam_data: dict) -> np.ndarray:
# # #         if "I2A" not in beam_data:
# # #             raise KeyError(
# # #                 f"Missing '{self.beam_key}.I2A' in TOML. This GUI requires a stored I2A matrix."
# # #             )
# # #         i2a = np.asarray(beam_data["I2A"], dtype=float)
# # #         if i2a.ndim != 2:
# # #             raise ValueError(f"Expected I2A to be 2D, got shape {i2a.shape}")
# # #         return i2a

# # #     def _infer_image_shape(self, i2a: np.ndarray, beam_data: dict, root_data: dict) -> tuple[int, int]:
# # #         n_pix = int(i2a.shape[1])
# # #         n_side = int(round(np.sqrt(n_pix)))

# # #         ctrl_model = (
# # #             beam_data.get(self.phasemask, {}).get("ctrl_model", {})
# # #             if isinstance(beam_data.get(self.phasemask, {}), dict)
# # #             else {}
# # #         )

# # #         for key in ("N0", "I0", "norm_pupil", "dark"):
# # #             if key in ctrl_model:
# # #                 arr = np.asarray(ctrl_model[key], dtype=float).reshape(-1)
# # #                 if arr.size == n_pix:
# # #                     return (n_side, n_side)

# # #         if n_side * n_side == n_pix:
# # #             return (n_side, n_side)

# # #         raise ValueError(
# # #             "Could not infer image shape from TOML. Please ensure I2A corresponds to a square ROI or add shape metadata."
# # #         )

# # #     def _load_reference_image(
# # #         self,
# # #         root_data: dict,
# # #         beam_data: dict,
# # #         image_shape: tuple[int, int],
# # #     ) -> Optional[np.ndarray]:
# # #         ctrl_model = beam_data.get(self.phasemask, {}).get("ctrl_model", {})
# # #         for key in ("N0", "I0", "norm_pupil", "dark"):
# # #             if key in ctrl_model:
# # #                 arr = np.asarray(ctrl_model[key], dtype=float).reshape(-1)
# # #                 if arr.size == image_shape[0] * image_shape[1]:
# # #                     return arr.reshape(image_shape)
# # #         return None


# # #     def _load_flat_command(self, root_data: dict, beam_data: dict) -> np.ndarray:
# # #         flat_cmd = np.zeros(140, dtype=float)

# # #         possible_paths = [
# # #             beam_data.get("dm_flat", None),
# # #             beam_data.get("DM_flat", None),
# # #             beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("dm_flat", None),
# # #             beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("DM_flat", None),
# # #         ]
# # #         for entry in possible_paths:
# # #             if entry is None:
# # #                 continue
# # #             if isinstance(entry, (str, bytes)):
# # #                 continue
# # #             try:
# # #                 arr = np.asarray(entry, dtype=float).reshape(-1)
# # #             except (TypeError, ValueError):
# # #                 continue
# # #             if arr.size == 140:
# # #                 flat_cmd = arr.copy()
# # #                 break
# # #             if arr.size == 144:
# # #                 flat_cmd = self._map_144_to_140(arr)
# # #                 break

# # #         return flat_cmd

# # #     # def _load_flat_command(self, root_data: dict, beam_data: dict) -> np.ndarray:
# # #     #     # Conservative default: zero command. If a flat becomes available later,
# # #     #     # it can be wired here without changing GUI logic.
# # #     #     flat_cmd = np.zeros(140, dtype=float)

# # #     #     # A few optional, harmless probes for future compatibility.
# # #     #     possible_paths = [
# # #     #         beam_data.get("dm_flat", None),
# # #     #         beam_data.get("DM_flat", None),
# # #     #         beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("dm_flat", None),
# # #     #         beam_data.get(self.phasemask, {}).get("ctrl_model", {}).get("DM_flat", None),
# # #     #     ]
# # #     #     for entry in possible_paths:
# # #     #         if entry is None:
# # #     #             continue
# # #     #         arr = np.asarray(entry, dtype=float).reshape(-1)
# # #     #         if arr.size == 140:
# # #     #             flat_cmd = arr.copy()
# # #     #             break
# # #     #         if arr.size == 144:
# # #     #             flat_cmd = self._map_144_to_140(arr)
# # #     #             break

# # #     #     return flat_cmd

# # #     def _load_m2c(self, root_data: dict, beam_data: dict) -> Optional[np.ndarray]:
# # #         ctrl_model = beam_data.get(self.phasemask, {}).get("ctrl_model", {})
# # #         if "M2C" not in ctrl_model:
# # #             return None
# # #         m2c = np.asarray(ctrl_model["M2C"], dtype=float)
# # #         if m2c.ndim != 2:
# # #             return None
# # #         return m2c

# # #     def _load_or_derive_actuator_positions(
# # #         self,
# # #         beam_data: dict,
# # #         i2a: np.ndarray,
# # #         image_shape: tuple[int, int],
# # #     ) -> np.ndarray:
# # #         # Future-compatible optional explicit keys.
# # #         explicit_keys = [
# # #             "actuator_coord_list_pixel_space",
# # #             "actuator_coords_pixel_space",
# # #             "actuator_pixel_coords",
# # #         ]
# # #         for key in explicit_keys:
# # #             if key in beam_data:
# # #                 arr = np.asarray(beam_data[key], dtype=float)
# # #                 if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] == i2a.shape[0]:
# # #                     return arr

# # #         # Otherwise derive from I2A row support.
# # #         return self._derive_positions_from_i2a(i2a, image_shape)

# # #     @staticmethod
# # #     def _derive_positions_from_i2a(i2a: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
# # #         ny, nx = image_shape
# # #         yy, xx = np.indices((ny, nx), dtype=float)
# # #         xx_flat = xx.reshape(-1)
# # #         yy_flat = yy.reshape(-1)

# # #         coords = np.zeros((i2a.shape[0], 2), dtype=float)
# # #         for k, row in enumerate(i2a):
# # #             weights = np.abs(np.asarray(row, dtype=float).reshape(-1))
# # #             total = float(np.sum(weights))
# # #             if not np.isfinite(total) or total <= 0:
# # #                 peak = int(np.argmax(np.abs(row)))
# # #                 coords[k, 0] = xx_flat[peak]
# # #                 coords[k, 1] = yy_flat[peak]
# # #                 continue
# # #             coords[k, 0] = np.sum(weights * xx_flat) / total
# # #             coords[k, 1] = np.sum(weights * yy_flat) / total
# # #         return coords

# # #     @staticmethod
# # #     def _map_144_to_140(cmd144: np.ndarray) -> np.ndarray:
# # #         grid = np.asarray(cmd144, dtype=float).reshape(12, 12)
# # #         keep = np.ones((12, 12), dtype=bool)
# # #         keep[0, 0] = False
# # #         keep[0, -1] = False
# # #         keep[-1, 0] = False
# # #         keep[-1, -1] = False
# # #         return grid[keep].reshape(-1)


# # # class ImageSourceBase(QtCore.QObject):
# # #     image_ready = QtCore.pyqtSignal(object)

# # #     def start(self) -> None:
# # #         raise NotImplementedError

# # #     def stop(self) -> None:
# # #         raise NotImplementedError

# # #     def latest(self) -> np.ndarray:
# # #         raise NotImplementedError


# # # class StaticImageSource(ImageSourceBase):
# # #     def __init__(self, shape: tuple[int, int], seed: int = 1234):
# # #         super().__init__()
# # #         rng = np.random.default_rng(seed)
# # #         self._image = rng.normal(size=shape).astype(float)

# # #     def start(self) -> None:
# # #         self.image_ready.emit(self._image.copy())

# # #     def stop(self) -> None:
# # #         pass

# # #     def latest(self) -> np.ndarray:
# # #         return self._image.copy()


# # # class LiveShmImageSource(ImageSourceBase):
# # #     def __init__(self, beam_id: int, timer_ms: int = 80):
# # #         super().__init__()
# # #         shm, _ = _import_live_backends()
# # #         self._shm = shm(f"/dev/shm/baldr{beam_id}.im.shm")
# # #         self._timer = QtCore.QTimer(self)
# # #         self._timer.setInterval(int(timer_ms))
# # #         self._timer.timeout.connect(self._poll)
# # #         self._latest: Optional[np.ndarray] = None

# # #     def _poll(self) -> None:
# # #         try:
# # #             img = np.asarray(self._shm.get_data(), dtype=float)
# # #             self._latest = img
# # #             self.image_ready.emit(img)
# # #         except Exception as exc:
# # #             print(f"Warning: failed to read SHM image: {exc}", file=sys.stderr)

# # #     def start(self) -> None:
# # #         self._poll()
# # #         self._timer.start()

# # #     def stop(self) -> None:
# # #         self._timer.stop()
# # #         try:
# # #             self._shm.close(erase_file=False)
# # #         except Exception:
# # #             pass

# # #     def latest(self) -> np.ndarray:
# # #         if self._latest is None:
# # #             raise RuntimeError("No live image has been received yet.")
# # #         return self._latest.copy()


# # # class DmSinkBase:
# # #     def send(self, cmd140: np.ndarray) -> None:
# # #         raise NotImplementedError

# # #     def close(self) -> None:
# # #         pass


# # # class NullDmSink(DmSinkBase):
# # #     def send(self, cmd140: np.ndarray) -> None:
# # #         return


# # # class LiveDmSink(DmSinkBase):
# # #     def __init__(self, beam_id: int):
# # #         _, dmclass = _import_live_backends()
# # #         self._dm = dmclass(beam_id=beam_id)

# # #     def send(self, cmd140: np.ndarray) -> None:
# # #         cmd140 = np.asarray(cmd140, dtype=float).reshape(-1)
# # #         if cmd140.size != 140:
# # #             raise ValueError(f"Expected 140-length DM command, got {cmd140.size}")
# # #         cmd2d = self._dm.cmd_2_map2D(cmd140, fill=0)
# # #         self._dm.set_data(cmd2d)


# # # @dataclass
# # # class ControllerState:
# # #     click_row: Optional[int] = None
# # #     click_col: Optional[int] = None
# # #     weights: Optional[np.ndarray] = None
# # #     amplitude: float = 0.0
# # #     preview_cmd: Optional[np.ndarray] = None


# # # class DmClickController(QtCore.QObject):
# # #     state_changed = QtCore.pyqtSignal()

# # #     def __init__(self, calib: CalibrationData, dm_sink: DmSinkBase):
# # #         super().__init__()
# # #         self.calib = calib
# # #         self.dm_sink = dm_sink
# # #         self.state = ControllerState()
# # #         self.send_enabled = False

# # #     def set_amplitude(self, value: float) -> None:
# # #         self.state.amplitude = float(value)
# # #         self._refresh_preview()

# # #     def set_click(self, row: int, col: int) -> None:
# # #         ny, nx = self.calib.image_shape
# # #         row = int(np.clip(row, 0, ny - 1))
# # #         col = int(np.clip(col, 0, nx - 1))
# # #         idx = row * nx + col
# # #         weights = np.asarray(self.calib.i2a[:, idx], dtype=float).reshape(-1)
# # #         self.state.click_row = row
# # #         self.state.click_col = col
# # #         self.state.weights = weights
# # #         self._refresh_preview()

# # #     def clear(self) -> None:
# # #         self.state = ControllerState(amplitude=self.state.amplitude)
# # #         self.state_changed.emit()

# # #     def zero_and_send(self) -> None:
# # #         self.state.preview_cmd = self.calib.flat_cmd.copy()
# # #         if self.send_enabled:
# # #             self.dm_sink.send(self.state.preview_cmd)
# # #         self.state_changed.emit()

# # #     def _refresh_preview(self) -> None:
# # #         if self.state.weights is None:
# # #             self.state.preview_cmd = self.calib.flat_cmd.copy()
# # #             self.state_changed.emit()
# # #             return
# # #         delta = self.state.amplitude * self.state.weights
# # #         self.state.preview_cmd = self.calib.flat_cmd + delta
# # #         if self.send_enabled:
# # #             self.dm_sink.send(self.state.preview_cmd)
# # #         self.state_changed.emit()

# # #     def top_contributors(self, n: int = 8) -> list[tuple[int, float]]:
# # #         if self.state.weights is None:
# # #             return []
# # #         w = np.asarray(self.state.weights, dtype=float).reshape(-1)
# # #         order = np.argsort(np.abs(w))[::-1][:n]
# # #         return [(int(i), float(w[i])) for i in order]


# # # class ClickableImageItem(pg.ImageItem):
# # #     clicked = QtCore.pyqtSignal(float, float)

# # #     def mouseClickEvent(self, ev):  # type: ignore[override]
# # #         if ev.button() == QtCore.Qt.LeftButton:
# # #             pos = ev.pos()
# # #             self.clicked.emit(float(pos.x()), float(pos.y()))
# # #             ev.accept()
# # #         else:
# # #             super().mouseClickEvent(ev)


# # # class MainWindow(QtWidgets.QMainWindow):
# # #     def __init__(
# # #         self,
# # #         calib: CalibrationData,
# # #         image_source: ImageSourceBase,
# # #         controller: DmClickController,
# # #         mode: str,
# # #         allow_send: bool,
# # #     ):
# # #         super().__init__()
# # #         self.calib = calib
# # #         self.image_source = image_source
# # #         self.controller = controller
# # #         self.mode = mode
# # #         self.allow_send = allow_send
# # #         self.current_image: Optional[np.ndarray] = None

# # #         self.setWindowTitle(f"DM Click GUI | beam {calib.beam_id} | {mode}")
# # #         self.resize(1300, 820)
# # #         self._build_ui()
# # #         self._connect_signals()

# # #         self.image_source.start()
# # #         self._update_status()
# # #         self._render_actuator_overlay()
# # #         self._update_dm_preview()

# # #     def _build_ui(self) -> None:
# # #         central = QtWidgets.QWidget()
# # #         self.setCentralWidget(central)
# # #         layout = QtWidgets.QHBoxLayout(central)

# # #         # Left: image view
# # #         self.graphics = pg.GraphicsLayoutWidget()
# # #         self.view = self.graphics.addViewBox(lockAspect=True, invertY=False)
# # #         self.view.setMouseEnabled(x=False, y=False)
# # #         self.image_item = ClickableImageItem(axisOrder="row-major")
# # #         self.view.addItem(self.image_item)

# # #         self.scatter_all = pg.ScatterPlotItem(size=10, pen=pg.mkPen("c", width=1.5), brush=None)
# # #         self.scatter_hot = pg.ScatterPlotItem(size=14, pen=pg.mkPen("r", width=2), brush=None)
# # #         self.click_marker = pg.ScatterPlotItem(size=12, pen=pg.mkPen("y", width=2), brush=None, symbol="x")
# # #         self.view.addItem(self.scatter_all)
# # #         self.view.addItem(self.scatter_hot)
# # #         self.view.addItem(self.click_marker)

# # #         # Right: controls and diagnostics
# # #         right = QtWidgets.QWidget()
# # #         right_layout = QtWidgets.QVBoxLayout(right)

# # #         mode_box = QtWidgets.QGroupBox("Mode")
# # #         mode_layout = QtWidgets.QFormLayout(mode_box)
# # #         self.mode_label = QtWidgets.QLabel(self.mode)
# # #         self.beam_label = QtWidgets.QLabel(str(self.calib.beam_id))
# # #         self.shape_label = QtWidgets.QLabel(f"{self.calib.image_shape[0]} x {self.calib.image_shape[1]}")
# # #         mode_layout.addRow("Mode:", self.mode_label)
# # #         mode_layout.addRow("Beam:", self.beam_label)
# # #         mode_layout.addRow("Image shape:", self.shape_label)
# # #         right_layout.addWidget(mode_box)

# # #         amp_box = QtWidgets.QGroupBox("Amplitude")
# # #         amp_layout = QtWidgets.QVBoxLayout(amp_box)
# # #         self.amp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
# # #         self.amp_slider.setRange(-1000, 1000)
# # #         self.amp_slider.setValue(0)
# # #         self.amp_slider.setTickInterval(100)
# # #         self.amp_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
# # #         self.amp_value_label = QtWidgets.QLabel("0.0000")
# # #         amp_layout.addWidget(self.amp_slider)
# # #         amp_layout.addWidget(self.amp_value_label)
# # #         right_layout.addWidget(amp_box)

# # #         send_box = QtWidgets.QGroupBox("Commanding")
# # #         send_layout = QtWidgets.QVBoxLayout(send_box)
# # #         self.send_checkbox = QtWidgets.QCheckBox("Enable live send")
# # #         self.send_checkbox.setChecked(False)
# # #         self.send_checkbox.setEnabled(self.mode == "live" and self.allow_send)
# # #         self.send_hint = QtWidgets.QLabel()
# # #         self.zero_button = QtWidgets.QPushButton("Zero / send flat")
# # #         self.clear_button = QtWidgets.QPushButton("Clear click")
# # #         send_layout.addWidget(self.send_checkbox)
# # #         send_layout.addWidget(self.send_hint)
# # #         send_layout.addWidget(self.zero_button)
# # #         send_layout.addWidget(self.clear_button)
# # #         right_layout.addWidget(send_box)

# # #         click_box = QtWidgets.QGroupBox("Click diagnostics")
# # #         click_layout = QtWidgets.QFormLayout(click_box)
# # #         self.pixel_label = QtWidgets.QLabel("-")
# # #         self.index_label = QtWidgets.QLabel("-")
# # #         self.norm_label = QtWidgets.QLabel("-")
# # #         self.maxw_label = QtWidgets.QLabel("-")
# # #         click_layout.addRow("Pixel (row, col):", self.pixel_label)
# # #         click_layout.addRow("Flat index:", self.index_label)
# # #         click_layout.addRow("||weights||:", self.norm_label)
# # #         click_layout.addRow("max |weight|:", self.maxw_label)
# # #         right_layout.addWidget(click_box)

# # #         contrib_box = QtWidgets.QGroupBox("Top actuator contributors")
# # #         contrib_layout = QtWidgets.QVBoxLayout(contrib_box)
# # #         self.contrib_text = QtWidgets.QPlainTextEdit()
# # #         self.contrib_text.setReadOnly(True)
# # #         self.contrib_text.setMinimumHeight(180)
# # #         contrib_layout.addWidget(self.contrib_text)
# # #         right_layout.addWidget(contrib_box)

# # #         dm_box = QtWidgets.QGroupBox("DM command preview (12x12)")
# # #         dm_layout = QtWidgets.QVBoxLayout(dm_box)
# # #         self.dm_view = pg.GraphicsLayoutWidget()
# # #         self.dm_plot = self.dm_view.addViewBox(lockAspect=True, invertY=False)
# # #         self.dm_img = pg.ImageItem(axisOrder="row-major")
# # #         self.dm_plot.addItem(self.dm_img)
# # #         dm_layout.addWidget(self.dm_view)
# # #         right_layout.addWidget(dm_box, stretch=1)

# # #         self.status_bar = QtWidgets.QLabel()
# # #         right_layout.addWidget(self.status_bar)
# # #         right_layout.addStretch(1)

# # #         layout.addWidget(self.graphics, stretch=3)
# # #         layout.addWidget(right, stretch=2)

# # #     def _connect_signals(self) -> None:
# # #         self.image_source.image_ready.connect(self._on_image)
# # #         self.image_item.clicked.connect(self._on_image_clicked)
# # #         self.amp_slider.valueChanged.connect(self._on_amp_slider)
# # #         self.send_checkbox.toggled.connect(self._on_send_toggled)
# # #         self.zero_button.clicked.connect(self.controller.zero_and_send)
# # #         self.clear_button.clicked.connect(self.controller.clear)
# # #         self.controller.state_changed.connect(self._on_state_changed)

# # #     def closeEvent(self, event):  # type: ignore[override]
# # #         try:
# # #             self.image_source.stop()
# # #         finally:
# # #             self.controller.dm_sink.close()
# # #         super().closeEvent(event)

# # #     def _amp_from_slider(self, value: int) -> float:
# # #         # Symmetric small range appropriate for safe DM perturbation tests.
# # #         return value / 10000.0

# # #     def _on_amp_slider(self, value: int) -> None:
# # #         amp = self._amp_from_slider(int(value))
# # #         self.amp_value_label.setText(f"{amp:+.4f}")
# # #         self.controller.set_amplitude(amp)

# # #     def _on_send_toggled(self, checked: bool) -> None:
# # #         self.controller.send_enabled = bool(checked and self.mode == "live" and self.allow_send)
# # #         self._update_status()
# # #         if self.controller.send_enabled:
# # #             # Push the current preview immediately when enabling.
# # #             self.controller.set_amplitude(self.controller.state.amplitude)

# # #     def _on_image(self, image: np.ndarray) -> None:
# # #         image = np.asarray(image, dtype=float)
# # #         if image.shape != self.calib.image_shape:
# # #             self.status_bar.setText(
# # #                 f"Warning: incoming image shape {image.shape} does not match calibration shape {self.calib.image_shape}"
# # #             )
# # #         self.current_image = image
# # #         self.image_item.setImage(image, autoLevels=True)
# # #         self.view.autoRange(padding=0.02)
# # #         self._render_click_marker()

# # #     def _on_image_clicked(self, x: float, y: float) -> None:
# # #         ny, nx = self.calib.image_shape
# # #         col = int(np.round(x))
# # #         row = int(np.round(y))
# # #         if not (0 <= row < ny and 0 <= col < nx):
# # #             return
# # #         self.controller.set_click(row=row, col=col)

# # #     def _on_state_changed(self) -> None:
# # #         self._render_click_marker()
# # #         self._render_hot_actuators()
# # #         self._update_click_diagnostics()
# # #         self._update_dm_preview()
# # #         self._update_status()

# # #     def _render_actuator_overlay(self) -> None:
# # #         xy = self.calib.actuator_xy_pix
# # #         spots = [{"pos": (float(x), float(y))} for x, y in xy]
# # #         self.scatter_all.setData(spots)

# # #     def _render_hot_actuators(self) -> None:
# # #         weights = self.controller.state.weights
# # #         if weights is None:
# # #             self.scatter_hot.setData([])
# # #             return

# # #         weights = np.asarray(weights, dtype=float).reshape(-1)
# # #         max_abs = float(np.max(np.abs(weights)))
# # #         if max_abs <= 0:
# # #             self.scatter_hot.setData([])
# # #             return

# # #         mask = np.abs(weights) >= 0.35 * max_abs
# # #         xy = self.calib.actuator_xy_pix[mask]
# # #         hot_weights = weights[mask]
# # #         sizes = 10 + 14 * (np.abs(hot_weights) / max_abs)
# # #         spots = [
# # #             {"pos": (float(x), float(y)), "size": float(s)}
# # #             for (x, y), s in zip(xy, sizes)
# # #         ]
# # #         self.scatter_hot.setData(spots)

# # #     def _render_click_marker(self) -> None:
# # #         row = self.controller.state.click_row
# # #         col = self.controller.state.click_col
# # #         if row is None or col is None:
# # #             self.click_marker.setData([])
# # #             return
# # #         self.click_marker.setData([{"pos": (float(col), float(row))}])

# # #     def _update_click_diagnostics(self) -> None:
# # #         row = self.controller.state.click_row
# # #         col = self.controller.state.click_col
# # #         weights = self.controller.state.weights
# # #         ny, nx = self.calib.image_shape

# # #         if row is None or col is None or weights is None:
# # #             self.pixel_label.setText("-")
# # #             self.index_label.setText("-")
# # #             self.norm_label.setText("-")
# # #             self.maxw_label.setText("-")
# # #             self.contrib_text.setPlainText("")
# # #             return

# # #         flat_idx = row * nx + col
# # #         self.pixel_label.setText(f"({row}, {col})")
# # #         self.index_label.setText(str(flat_idx))
# # #         self.norm_label.setText(f"{np.linalg.norm(weights):.4e}")
# # #         self.maxw_label.setText(f"{np.max(np.abs(weights)):.4e}")

# # #         lines = []
# # #         for act, val in self.controller.top_contributors(n=10):
# # #             lines.append(f"act {act:3d}: {val:+.6e}")
# # #         self.contrib_text.setPlainText("\n".join(lines))

# # #     def _update_dm_preview(self) -> None:
# # #         cmd = self.controller.state.preview_cmd
# # #         if cmd is None:
# # #             cmd = self.calib.flat_cmd
# # #         grid = self._cmd140_to_12x12(np.asarray(cmd, dtype=float).reshape(-1))
# # #         self.dm_img.setImage(grid, autoLevels=True)
# # #         self.dm_plot.autoRange(padding=0.04)

# # #     def _update_status(self) -> None:
# # #         if self.mode == "static":
# # #             self.send_hint.setText("Static mode: hardware writes disabled.")
# # #         elif self.mode == "live-readonly":
# # #             self.send_hint.setText("Live-readonly mode: camera live, DM writes disabled.")
# # #         elif self.mode == "live" and not self.allow_send:
# # #             self.send_hint.setText("Live mode without --allow-send: DM writes disabled.")
# # #         elif self.mode == "live" and self.allow_send and not self.controller.send_enabled:
# # #             self.send_hint.setText("Live mode: enable checkbox to send commands.")
# # #         else:
# # #             self.send_hint.setText("Live mode: DM sending enabled.")

# # #         mode_txt = self.mode
# # #         send_txt = "ON" if self.controller.send_enabled else "OFF"
# # #         self.status_bar.setText(f"Mode={mode_txt} | Send={send_txt} | Amplitude={self.controller.state.amplitude:+.4f}")

# # #     @staticmethod
# # #     def _cmd140_to_12x12(cmd140: np.ndarray) -> np.ndarray:
# # #         cmd140 = np.asarray(cmd140, dtype=float).reshape(-1)
# # #         if cmd140.size != 140:
# # #             raise ValueError(f"Expected 140-length command, got {cmd140.size}")
# # #         out = np.full((12, 12), np.nan, dtype=float)
# # #         keep = np.ones((12, 12), dtype=bool)
# # #         keep[0, 0] = False
# # #         keep[0, -1] = False
# # #         keep[-1, 0] = False
# # #         keep[-1, -1] = False
# # #         out[keep] = cmd140
# # #         return out


# # # def build_argparser() -> argparse.ArgumentParser:
# # #     parser = argparse.ArgumentParser(description="Interactive click-to-DM GUI using stored Baldr I2A calibration.")
# # #     parser.add_argument("--toml", required=True, help="Path to beam TOML config file")
# # #     parser.add_argument("--beam", type=int, required=True, help="Beam ID")
# # #     parser.add_argument("--phasemask", default="H4", help="Phasemask key under beam{N} in TOML")
# # #     parser.add_argument(
# # #         "--mode",
# # #         choices=["static", "live-readonly", "live"],
# # #         default="static",
# # #         help="GUI operating mode",
# # #     )
# # #     parser.add_argument(
# # #         "--allow-send",
# # #         action="store_true",
# # #         help="Allow DM commands in live mode. Even with this flag, live sending remains disabled until the GUI checkbox is enabled.",
# # #     )
# # #     return parser


# # # def main() -> int:
# # #     parser = build_argparser()
# # #     args = parser.parse_args()

# # #     loader = TomlCalibrationLoader(toml_path=args.toml, beam_id=args.beam, phasemask=args.phasemask)
# # #     calib = loader.load()

# # #     if args.mode == "static":
# # #         image_source = StaticImageSource(shape=calib.image_shape)
# # #         dm_sink: DmSinkBase = NullDmSink()
# # #     elif args.mode == "live-readonly":
# # #         image_source = LiveShmImageSource(beam_id=calib.beam_id)
# # #         dm_sink = NullDmSink()
# # #     elif args.mode == "live":
# # #         image_source = LiveShmImageSource(beam_id=calib.beam_id)
# # #         dm_sink = LiveDmSink(beam_id=calib.beam_id) if args.allow_send else NullDmSink()
# # #     else:  # pragma: no cover
# # #         raise ValueError(f"Unsupported mode: {args.mode}")

# # #     app = QtWidgets.QApplication(sys.argv)
# # #     pg.setConfigOptions(imageAxisOrder="row-major")
# # #     controller = DmClickController(calib=calib, dm_sink=dm_sink)
# # #     window = MainWindow(
# # #         calib=calib,
# # #         image_source=image_source,
# # #         controller=controller,
# # #         mode=args.mode,
# # #         allow_send=bool(args.allow_send),
# # #     )
# # #     window.show()
# # #     return app.exec_()


# # # if __name__ == "__main__":
# # #     raise SystemExit(main())
