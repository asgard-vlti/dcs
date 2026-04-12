import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import zmq
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from LazyPirateZMQ import ZmqLazyPirateClient
from consts import BEAM_TO_PORT

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib


GAIN_STEP = 0.005
GAIN_MIN = 0.0
GAIN_MAX = 0.2
GAIN_TICKS = int(round((GAIN_MAX - GAIN_MIN) / GAIN_STEP))


@dataclass
class ModeBlock:
    name: str
    idx_spec: str
    gain_value: float = 0.0
    leak_value: float = 0.0


class CommandSender:
    def __init__(self, beam: int, simulation: bool):
        self.simulation = simulation
        self.client: Optional[ZmqLazyPirateClient] = None
        self.context: Optional[zmq.Context] = None

        if not self.simulation:
            endpoint = f"tcp://mimir:{BEAM_TO_PORT[beam]}"
            self.context = zmq.Context.instance()
            self.client = ZmqLazyPirateClient(self.context, endpoint)

    def send(self, command: str) -> str:
        if self.simulation:
            print(command)
            return "SIM"
        if self.client is None:
            raise RuntimeError("Command client is not initialised")
        return self.client.send_and_recv(command)

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None


class GainLeakWindow(QMainWindow):
    def __init__(
        self,
        blocks: list[ModeBlock],
        sender: CommandSender,
        title: Optional[str] = None,
    ):
        super().__init__()
        self.blocks = blocks
        self.sender = sender
        if title:
            self.setWindowTitle(title)
        else:
            self.setWindowTitle(f"Baldr AO Control")
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        sliders_layout = QHBoxLayout()
        main_layout.addLayout(sliders_layout)

        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        for block in self.blocks:
            block_widget = self._create_block_widget(block)
            sliders_layout.addWidget(block_widget)

        self._initialise_system_defaults()

        central.setLayout(main_layout)

    def _create_block_widget(self, block: ModeBlock) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout()

        name_label = QLabel(block.name)
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)

        gain_value_label = QLabel(f"{block.gain_value:.3f}")
        gain_value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(gain_value_label)

        slider = QSlider(Qt.Vertical)
        slider.setMinimum(0)
        slider.setMaximum(GAIN_TICKS)
        slider.setTickPosition(QSlider.TicksRight)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        slider.setValue(self._gain_to_tick(block.gain_value))
        slider.valueChanged.connect(
            lambda tick, b=block, lbl=gain_value_label: self._on_gain_changed(
                b, tick, lbl
            )
        )
        layout.addWidget(slider)

        leak_edit = QLineEdit(f"{block.leak_value:.3f}")
        leak_edit.setPlaceholderText("Leak")
        leak_edit.editingFinished.connect(
            lambda b=block, edit=leak_edit: self._on_leak_updated(b, edit)
        )
        layout.addWidget(leak_edit)

        widget.setLayout(layout)
        return widget

    @staticmethod
    def _gain_to_tick(gain_value: float) -> int:
        return int(round((gain_value - GAIN_MIN) / GAIN_STEP))

    def _initialise_system_defaults(self):
        for block in self.blocks:
            gain_text = f"{block.gain_value:.3f}"
            leak_text = f"{block.leak_value:.3f}"
            self._send_command(f"set_ki_gains {block.idx_spec} {gain_text}")
            self._send_command(f"set_leaks {block.idx_spec} {leak_text}")

    def _on_gain_changed(self, block: ModeBlock, tick: int, label: QLabel):
        gain_value = GAIN_MIN + tick * GAIN_STEP
        gain_text = f"{gain_value:.3f}"
        label.setText(gain_text)
        command = f"set_ki_gains {block.idx_spec} {gain_text}"
        self._send_command(command)

    def _on_leak_updated(self, block: ModeBlock, leak_edit: QLineEdit):
        raw_text = leak_edit.text().strip()
        try:
            leak_value = float(raw_text)
        except ValueError:
            leak_edit.setText(f"{block.leak_value:.3f}")
            self.status_label.setText(
                f"Invalid leak value for {block.name}: '{raw_text}'"
            )
            return

        block.leak_value = leak_value
        leak_text = f"{leak_value:.3f}"
        leak_edit.setText(leak_text)
        command = f"set_leaks {block.idx_spec} {leak_text}"
        self._send_command(command)

    def _send_command(self, command: str):
        try:
            response = self.sender.send(command)
            self.status_label.setText(f"{command} -> {response}")
        except Exception as exc:
            self.status_label.setText(f"Command failed: {exc}")


def load_blocks(config_path: Path) -> list[ModeBlock]:
    with config_path.open("rb") as f:
        data = tomllib.load(f)

    blocks_table = data.get("blocks")
    if not isinstance(blocks_table, dict) or not blocks_table:
        raise ValueError("Config file must define a non-empty [blocks] table")

    defaults = data.get("default", {})
    if not isinstance(defaults, dict):
        raise ValueError("Config [default] must be a table")

    default_gain_raw = defaults.get("gain", GAIN_MIN)
    default_leak_raw = defaults.get("leak", 0.0)
    try:
        default_gain = float(default_gain_raw)
        default_leak = float(default_leak_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Default gain/leak must be numeric") from exc

    if default_gain < GAIN_MIN or default_gain > GAIN_MAX:
        raise ValueError(
            f"Default gain must be within [{GAIN_MIN:.3f}, {GAIN_MAX:.3f}]"
        )

    # Enforce exact slider-representable gain increments.
    default_gain = round(default_gain / GAIN_STEP) * GAIN_STEP

    blocks: list[ModeBlock] = []
    for name, idx_spec in blocks_table.items():
        if not isinstance(idx_spec, str):
            raise ValueError(f"Block '{name}' index spec must be a string")
        blocks.append(
            ModeBlock(
                name=name,
                idx_spec=idx_spec.strip(),
                gain_value=default_gain,
                leak_value=default_leak,
            )
        )
    return blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Baldr AO gain/leak GUI")
    parser.add_argument(
        "--beam",
        type=int,
        required=True,
        choices=sorted(BEAM_TO_PORT.keys()),
        help="Beam number to control",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Simulation mode: print commands instead of sending them",
    )
    parser.add_argument(
        "--conf",
        type=Path,
        default=Path(__file__).with_name("gui_conf.toml"),
        help="Path to GUI config TOML",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.conf.exists():
        raise FileNotFoundError(f"Config file not found: {args.conf}")

    blocks = load_blocks(args.conf)
    sender = CommandSender(beam=args.beam, simulation=args.sim)

    app = QApplication([])
    app.aboutToQuit.connect(sender.close)

    window = GainLeakWindow(
        blocks=blocks, sender=sender, title=f"Baldr AO Control - Beam {args.beam}"
    )
    window.resize(400, 500)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
