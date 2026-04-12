import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import zmq
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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
DEFAULT_OPEN_THRESHOLD = 0.7
DEFAULT_CLOSE_THRESHOLD = 0.5


@dataclass
class ModeBlock:
    name: str
    idx_spec: str
    gain_value: float = 0.0
    leak_value: float = 0.0


@dataclass
class GuiDefaults:
    open_threshold: float
    close_threshold: float
    servo_on: bool


@dataclass
class GuiConfig:
    blocks: list[ModeBlock]
    defaults: GuiDefaults


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
        defaults: GuiDefaults,
        sender: CommandSender,
        title: Optional[str] = None,
    ):
        super().__init__()
        self.blocks = blocks
        self.defaults = defaults
        self.cmd_sender = sender
        if title:
            self.setWindowTitle(title)
        else:
            self.setWindowTitle("Baldr AO Control")
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()

        controls_layout = QHBoxLayout()
        self.servo_checkbox = QCheckBox("Servo")
        self.servo_checkbox.setChecked(self.defaults.servo_on)
        self.servo_checkbox.stateChanged.connect(self._on_servo_toggled)
        controls_layout.addWidget(self.servo_checkbox)

        self.close_thresh_edit = self._create_numeric_command_edit(
            command_name="set_close_threshold",
            default_value=self.defaults.close_threshold,
            error_context="close threshold",
            scientific_notation=True,
        )
        controls_layout.addWidget(QLabel("Close"))
        controls_layout.addWidget(self.close_thresh_edit)

        self.open_thresh_edit = self._create_numeric_command_edit(
            command_name="set_open_threshold",
            default_value=self.defaults.open_threshold,
            error_context="open threshold",
        )
        controls_layout.addWidget(QLabel("Open"))
        controls_layout.addWidget(self.open_thresh_edit)
        controls_layout.addStretch(1)
        main_layout.addLayout(controls_layout)

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

    def _create_numeric_command_edit(
        self,
        command_name: str,
        default_value: float,
        error_context: str,
        scientific_notation: bool = False,
    ) -> QLineEdit:
        edit = QLineEdit(self._format_threshold(default_value, scientific_notation))
        edit.setMinimumWidth(70)

        def _handle_editing_finished(
            e=edit,
            cmd=command_name,
            ctx=error_context,
            sci=scientific_notation,
        ):
            self._on_numeric_command_updated(e, cmd, ctx, sci)

        edit.editingFinished.connect(_handle_editing_finished)
        return edit

    @staticmethod
    def _format_threshold(value: float, scientific_notation: bool) -> str:
        if scientific_notation:
            return f"{value:.3e}"
        return f"{value:.3f}"

    def _on_numeric_command_updated(
        self,
        edit: QLineEdit,
        command_name: str,
        error_context: str,
        scientific_notation: bool = False,
    ):
        raw_text = edit.text().strip()
        try:
            value = float(raw_text)
        except ValueError:
            self.status_label.setText(f"Invalid {error_context}: '{raw_text}'")
            return

        display_text = self._format_threshold(value, scientific_notation)
        edit.setText(display_text)
        self._send_command(f"{command_name} {value:.3f}")

    def _on_servo_toggled(self, state: int):
        servo_state = "on" if state == Qt.Checked else "off"
        self._send_command(f"servo {servo_state}")

    def _initialise_system_defaults(self):
        servo_state = "on" if self.defaults.servo_on else "off"
        self._send_command(f"servo {servo_state}")
        self._send_command(f"set_close_threshold {self.defaults.close_threshold:.3f}")
        self._send_command(f"set_open_threshold {self.defaults.open_threshold:.3f}")
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
            response = self.cmd_sender.send(command)
            self.status_label.setText(f"{command} -> {response}")
        except Exception as exc:
            self.status_label.setText(f"Command failed: {exc}")


def _parse_float_default(defaults: dict, key: str, fallback: float) -> float:
    raw_value = defaults.get(key, fallback)
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Default '{key}' must be numeric") from exc


def _parse_bool_default(defaults: dict, key: str, fallback: bool) -> bool:
    raw_value = defaults.get(key, fallback)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        lowered = raw_value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Default '{key}' must be a boolean")


def load_config(config_path: Path) -> GuiConfig:
    with config_path.open("rb") as f:
        data = tomllib.load(f)

    blocks_table = data.get("blocks")
    if not isinstance(blocks_table, dict) or not blocks_table:
        raise ValueError("Config file must define a non-empty [blocks] table")

    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("Config [defaults] must be a table")

    default_open_threshold = _parse_float_default(
        defaults, "open_threshold", DEFAULT_OPEN_THRESHOLD
    )
    default_close_threshold = _parse_float_default(
        defaults, "close_threshold", DEFAULT_CLOSE_THRESHOLD
    )
    default_servo_on = _parse_bool_default(defaults, "servo_on", False)

    blocks: list[ModeBlock] = []
    for name, idx_spec in blocks_table.items():
        if not isinstance(idx_spec, str):
            raise ValueError(f"Block '{name}' index spec must be a string")

        block_defaults = defaults.get(name, {})
        if block_defaults is None:
            block_defaults = {}
        if not isinstance(block_defaults, dict):
            raise ValueError(f"Defaults for block '{name}' must be a table")

        block_gain = _parse_float_default(block_defaults, "gain", GAIN_MIN)
        block_leak = _parse_float_default(block_defaults, "leak", 0.0)

        if block_gain < GAIN_MIN or block_gain > GAIN_MAX:
            raise ValueError(
                f"Default gain for block '{name}' must be within [{GAIN_MIN:.3f}, {GAIN_MAX:.3f}]"
            )

        # Enforce exact slider-representable gain increments.
        block_gain = round(block_gain / GAIN_STEP) * GAIN_STEP

        blocks.append(
            ModeBlock(
                name=name,
                idx_spec=idx_spec.strip(),
                gain_value=block_gain,
                leak_value=block_leak,
            )
        )

    return GuiConfig(
        blocks=blocks,
        defaults=GuiDefaults(
            open_threshold=default_open_threshold,
            close_threshold=default_close_threshold,
            servo_on=default_servo_on,
        ),
    )


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

    config = load_config(args.conf)
    sender = CommandSender(beam=args.beam, simulation=args.sim)

    app = QApplication([])
    app.aboutToQuit.connect(sender.close)

    window = GainLeakWindow(
        blocks=config.blocks,
        defaults=config.defaults,
        sender=sender,
        title=f"Baldr AO Control - Beam {args.beam}",
    )
    window.resize(400, 500)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
