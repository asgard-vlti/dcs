"""
dmview.py

A simple viewer for the DM SHM data, with no buttons. Shows the total command on the left
and all individual channels to the right in a single row.

Usage:
    python dmview.py <beam>
    python dmview.py -1

"""

import argparse
import pathlib
import sys
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from xaosim.shmlib import shm

Frame = np.ndarray


class DMshm:
    VMIN = 0
    VMAX = 1

    ARR_SHAPE = (12, 12)

    N_CHANNELS = 5

    BASE_PTH = pathlib.Path("/dev/shm/")

    def __init__(self, beam):
        total_fname = f"dm{beam}.im.shm"
        self.total_s = shm(self.BASE_PTH / total_fname)
        self.channel_s = []
        for i in range(self.N_CHANNELS):
            channel_fname = f"dm{beam}disp{i:02d}.im.shm"
            self.channel_s.append(shm(self.BASE_PTH / channel_fname))
            print(f"Opened shm to: {self.BASE_PTH / channel_fname}")

    def _read_frame(self, shm_obj) -> Optional[Frame]:
        try:
            data = shm_obj.get_data()
        except (OSError, RuntimeError, ValueError, AttributeError):
            return None

        if data is None:
            return None

        arr = np.asarray(data)
        if arr.size != self.ARR_SHAPE[0] * self.ARR_SHAPE[1]:
            return None

        return arr.reshape(self.ARR_SHAPE)

    def update(self):
        """
        Read total + channel frames from shared memory.

        Returns:
            tuple[np.ndarray | None, list[np.ndarray | None]]: total frame and channel frames.
        """

        total = self._read_frame(self.total_s)
        # channels = [self._read_frame(ch) for ch in self.channel_s]
        channels = []
        for i, ch in enumerate(
            self.channel_s
        ):  # channels 1-4 need to be scaled to be in [0,1] (currently [-0.5,0.5])
            if i == 0:
                channels.append(self._read_frame(ch))
            else:
                frame = self._read_frame(ch)
                if frame is not None:
                    frame = (frame + 0.5).clip(0.0, 1.0)
                channels.append(frame)
        return total, channels


def _build_cmap_lut(name: str) -> np.ndarray:
    """Build a 256-color cmap LUT. Falls back to a fixed blue-red LUT if needed."""
    try:
        import matplotlib

        # Matplotlib 3.7+: use the non-deprecated colormap registry API.
        if hasattr(matplotlib, "colormaps"):
            cmap = matplotlib.colormaps[name]
        else:
            from matplotlib import cm

            cmap = cm.get_cmap(name)

        rgba = cmap(np.linspace(0.0, 1.0, 256))
        return (rgba[:, :3] * 255).astype(np.uint8)
    except ImportError:
        ramp = np.linspace(0.0, 1.0, 256)
        red = (255 * ramp).astype(np.uint8)
        blue = (255 * (1.0 - ramp)).astype(np.uint8)
        green = np.zeros_like(red, dtype=np.uint8)
        return np.column_stack((red, green, blue))


class DMView(QtWidgets.QMainWindow):
    UPDATE_MS = 200  # 5 Hz refresh for low bandwidth and stable CPU usage.
    SATURATION_EPS = 0.01

    class _BeamRow:
        def __init__(self, beam: str):
            self.beam = beam
            self.dm = DMshm(beam)
            self.image_items: list[pg.ImageItem] = []
            self.last_frames: list[Optional[Frame]] = [None] * (self.dm.N_CHANNELS + 1)
            self.total_overlay: Optional[pg.ImageItem] = None

    @staticmethod
    def _frame_levels(frame: Frame) -> tuple[float, float]:
        low = float(np.nanmin(frame))
        high = float(np.nanmax(frame))

        if not np.isfinite(low) or not np.isfinite(high):
            return 0.0, 1.0

        if high <= low:
            pad = 0.05 if low == 0.0 else abs(low) * 0.05
            return low - pad, high + pad

        span = high - low
        pad = span * 0.02
        return low - pad, high + pad

    def __init__(self, beams):
        super().__init__()
        beam_list = [str(beam) for beam in beams]
        self.setWindowTitle(f"DM SHM Viewer - beam {', '.join(beam_list)}")

        self._cividis_lut = _build_cmap_lut("cividis")
        self._rdbu_lut = _build_cmap_lut("RdBu")

        central = pg.GraphicsLayoutWidget()
        self.setCentralWidget(central)
        central.ci.setSpacing(2)
        central.ci.setContentsMargins(2, 2, 2, 2)

        self._labels = ["total"] + [f"ch{i:02d}" for i in range(DMshm.N_CHANNELS)]
        self._rows: list[DMView._BeamRow] = []

        for row_idx, beam in enumerate(beam_list):
            row = self._BeamRow(beam)
            self._rows.append(row)

            for col_idx, label in enumerate(self._labels):
                title = label if len(beam_list) == 1 or row_idx == 0 else None
                plot = central.addPlot(row=row_idx, col=col_idx, title=title)
                plot.setAspectLocked(True)
                plot.setContentsMargins(0, 0, 0, 0)
                plot.hideAxis("left")
                plot.hideAxis("bottom")

                image_item = pg.ImageItem(axisOrder="row-major")
                image_item.setLookupTable(
                    self._cividis_lut if col_idx < 2 else self._rdbu_lut
                )
                image_item.setLevels((row.dm.VMIN, row.dm.VMAX))
                plot.addItem(image_item)
                row.image_items.append(image_item)

                if col_idx == 0:
                    row.total_overlay = pg.ImageItem(axisOrder="row-major")
                    row.total_overlay.setZValue(10)
                    plot.addItem(row.total_overlay)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(self.UPDATE_MS)

    def _total_saturation_overlay(self, frame: Frame) -> np.ndarray:
        overlay = np.zeros((*frame.shape, 4), dtype=np.uint8)
        low = self.dm.VMIN + self.SATURATION_EPS
        high = self.dm.VMAX - self.SATURATION_EPS
        mask = (frame <= low) | (frame >= high)
        mask[0, 0] = False
        mask[0, -1] = False
        mask[-1, 0] = False
        mask[-1, -1] = False
        overlay[mask] = (255, 0, 0, 255)
        return overlay

    def _refresh(self):
        for row in self._rows:
            total, channels = row.dm.update()
            frames = [total] + channels

            for i, frame in enumerate(frames):
                if frame is None:
                    frame = row.last_frames[i]
                else:
                    row.last_frames[i] = frame

                if frame is not None:
                    row.image_items[i].setLevels(self._frame_levels(frame))
                    row.image_items[i].setImage(frame, autoLevels=False)

                    if i == 0 and row.total_overlay is not None:
                        row.total_overlay.setImage(
                            self._total_saturation_overlay(frame), autoLevels=False
                        )


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Low-bandwidth DM SHM viewer")
    parser.add_argument(
        "beam",
        nargs="+",
        help="One or more beam indices used in dm<beam> shared-memory names",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    beams = [str(beam) for beam in range(1, 5)] if args.beam == ["-1"] else args.beam

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    class GlobalHotkeyFilter(QtCore.QObject):
        def eventFilter(self, a0, a1):
            if isinstance(a1, QtGui.QKeyEvent) and a1.matches(
                QtGui.QKeySequence.Cancel
            ):
                for widget in QtWidgets.QApplication.topLevelWidgets():
                    widget.close()
                QtCore.QCoreApplication.quit()
                return True
            return super().eventFilter(a0, a1)

    hotkey_filter = GlobalHotkeyFilter(app)
    app.installEventFilter(hotkey_filter)

    win = DMView(beams)
    win.resize(1100, 250 * max(1, len(beams)))
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
