"""Numerical parity with the working-tree toy-interferometer/main2.py."""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as onp


ROOT = Path(__file__).resolve().parents[3]
TOY = Path(os.environ.get("HEIMDALLR_TOY_DIR", ROOT / "toy-interferometer"))
SOURCE = Path(__file__).with_name("predictive_driver.cpp")


class PredictiveParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        sys.path.insert(0, str(TOY))
        spec = importlib.util.spec_from_file_location("main2", TOY / "main2.py")
        cls.main2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.main2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.executable = Path(cls.temp.name) / "predictive_driver"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-O3",
                "-DNDEBUG",
                "-I/usr/include/eigen3",
                str(SOURCE),
                str(SOURCE.parents[1] / "predictive_control.cpp"),
                "-o",
                str(cls.executable),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_driver(self, payload):
        result = subprocess.run(
            [str(self.executable)],
            input=payload,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.splitlines()

    def compare_controller(self, history, future, count):
        rng = onp.random.default_rng(20260930)
        controller = self.main2.PredictiveControl(
            n_actuators=3,
            n_history=history,
            n_future=future,
            n_explore=500,
            gain=0.2,
            learning_exponent=1.0,
            initial_regularization=1e5,
            initial_covariance=1e-6,
            method="QRD",
            explore_sigma=0.01,
            explore_max=0.4,
            delta_max=0.5,
        )
        inputs = [f"controller {history} {future} {count}"]
        expected_commands = []
        for i in range(count):
            error = rng.normal(0.0, 0.06, size=3)
            draw = rng.normal(size=3)
            if i == 10:
                draw = onp.array([100.0, -100.0, 0.0])
            if i % 100 == 0 and controller.regularization[0, 0] > 0.003:
                controller.regularization /= 5
            with patch.object(self.main2.np.random, "randn", return_value=draw):
                proposed = controller.propose_command(error).copy()
            applied = onp.clip(proposed, -0.25, 0.25)
            controller.update(applied)
            controller.u = applied.copy()
            expected_commands.append(proposed)
            inputs.append(" ".join(map(str, onp.r_[error, draw, applied])))

        lines = self.run_driver("\n".join(inputs) + "\n")
        actual_commands = onp.array(
            [[float(x) for x in line.split()] for line in lines[:count]]
        )
        onp.testing.assert_allclose(
            actual_commands, expected_commands, rtol=1e-9, atol=1e-10
        )
        self.assertAlmostEqual(float(lines[count].split()[1]), controller.regularization[0, 0])

        def state(name, shape):
            label, *values = lines.pop(count + 1).split()
            self.assertEqual(label, name)
            return onp.asarray(values, dtype=float).reshape(shape)

        weight = state("WEIGHTS", controller._rls.W.shape)
        gram = state("GRAM", (controller.n_features, controller.n_features))
        predictive = state("PREDICTIVE", controller.predictive_controller.shape)
        onp.testing.assert_allclose(weight, controller._rls.W, rtol=1e-9, atol=1e-10)
        onp.testing.assert_allclose(
            gram, controller._rls.R.T @ controller._rls.R, rtol=1e-9, atol=1e-10
        )
        onp.testing.assert_allclose(
            predictive, controller.predictive_controller, rtol=1e-9, atol=1e-10
        )

    def test_history_regularization_exploration_and_saturation(self):
        self.compare_controller(history=4, future=2, count=510)

    def test_production_dimensions_and_fresh_model(self):
        self.compare_controller(history=40, future=4, count=110)
        self.compare_controller(history=40, future=4, count=51)

    def test_phase_sign_and_dm_units(self):
        wavelength = 2.1
        phase = onp.array([0.05, -0.02, 0.01, -0.04])
        applied_dm = onp.array([0.4, -0.4, 0.1, -0.1])
        lines = self.run_driver(
            "transform "
            + " ".join(map(str, onp.r_[wavelength, phase, applied_dm]))
            + "\n"
        )
        error = self.main2.DM2S @ -phase
        onp.testing.assert_allclose(onp.fromstring(lines[0], sep=" "), error, atol=1e-10)
        onp.testing.assert_allclose(
            onp.fromstring(lines[1], sep=" "),
            self.main2.S2DM @ error * wavelength / 6.0,
            atol=1e-10,
        )
        onp.testing.assert_allclose(
            onp.fromstring(lines[2], sep=" "),
            self.main2.DM2S @ (applied_dm * 6.0 / wavelength),
            atol=1e-10,
        )

    def test_mode_entry_preserves_command_and_rearms_exploration(self):
        initial_dm = onp.array([0.1, 0.05, 0.0, -0.05])
        lines = self.run_driver(
            "servo " + " ".join(map(str, onp.r_[2.1, 6.0, 0.4, initial_dm])) + "\n"
        )
        first, after_loss, after_entry = (
            onp.fromstring(line, sep=" ") for line in lines
        )
        onp.testing.assert_allclose(first, initial_dm, rtol=1e-9, atol=1e-10)
        onp.testing.assert_allclose(after_loss, initial_dm, rtol=1e-9, atol=1e-10)
        self.assertGreater(onp.max(onp.abs(after_entry - initial_dm)), 1e-3)

    def test_servo_command_and_applied_feedback_match_python(self):
        rng = onp.random.default_rng(44)
        initial_dm = onp.array([0.38, -0.38, 0.36, -0.36])
        wavelength = 2.1
        controller = self.main2.PredictiveControl(
            n_actuators=3,
            n_history=40,
            n_future=4,
            n_explore=500,
            gain=0.2,
            learning_exponent=1.0,
            initial_regularization=1e5,
            initial_covariance=1e-6,
            method="QRD",
            explore_sigma=0.01,
            explore_max=0.4,
            delta_max=0.5,
        )
        controller.u = self.main2.DM2S @ (initial_dm * 6.0 / wavelength)
        common = initial_dm.mean()
        inputs = [
            "servo_trace 110 "
            + " ".join(map(str, onp.r_[wavelength, 6.0, 0.4, initial_dm]))
        ]
        expected = []
        saturated = False
        for i in range(110):
            phase = rng.normal(0.0, 0.3, size=4)
            draw = rng.normal(size=3)
            if i % 100 == 0 and controller.regularization[0, 0] > 0.003:
                controller.regularization /= 5
            with patch.object(self.main2.np.random, "randn", return_value=draw):
                modes = controller.propose_command(self.main2.DM2S @ -phase)
            proposed = self.main2.S2DM @ modes * wavelength / 6.0 + common
            applied = onp.clip(proposed, -0.4, 0.4)
            saturated |= bool(onp.any(applied != proposed))
            applied_modes = self.main2.DM2S @ (applied * 6.0 / wavelength)
            controller.update(applied_modes)
            controller.u = applied_modes.copy()
            expected.append(applied)
            inputs.append(" ".join(map(str, onp.r_[phase, draw])))
        actual = onp.array(
            [[float(x) for x in line.split()] for line in self.run_driver("\n".join(inputs) + "\n")]
        )
        self.assertTrue(saturated)
        onp.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
