"""Offline checks for flattening sources; never connect to instrument hardware."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as onp
from astropy.io import fits


SCRIPT = Path(__file__).resolve().parents[1] / "dcs/cmd_scripts/flatten_beam.py"
spec = importlib.util.spec_from_file_location("flatten_beam", SCRIPT)
flatten = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flatten)


class FlattenSourcesTest(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.root = Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        self.context.enter_context(patch.object(flatten.Path, "home", return_value=self.root))
        self.context.enter_context(patch.object(flatten, "USE_FITS", True))
        self.context.enter_context(redirect_stdout(io.StringIO()))
        self.context.enter_context(redirect_stderr(io.StringIO()))
        self.clear = onp.arange(1024, dtype=float).reshape(32, 32) + 10
        self.reference = self.clear[::-1].copy()

    def source_path(self, directory, beam=1, suffix=".fits"):
        path = self.root / "etc" / directory / f"beam{beam}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_pair(self, directory, beam=1):
        path = self.source_path(directory, beam)
        # Reverse HDU order so extension selection cannot pass by accident.
        primary = fits.PrimaryHDU(self.reference)
        primary.header["EXTNAME"] = "PHASE_MASK"
        fits.HDUList([
            primary,
            fits.ImageHDU(self.clear, name="CLEAR_PUPIL"),
        ]).writeto(path, overwrite=True)
        return path

    def test_cli_defaults_and_source_combinations(self):
        args = flatten.parse_args(["1"])
        self.assertEqual((args.source, args.target), ("live", "model"))
        for source in ("live", "saved-pupil", "saved-ref"):
            for target in ("stddev", "model", "amp-model"):
                with self.subTest(source=source, target=target):
                    argv = ["1", "--source", source, "--target", target]
                    if (source, target) == ("saved-ref", "stddev"):
                        with self.assertRaises(SystemExit) as error:
                            flatten.parse_args(argv)
                        self.assertEqual(error.exception.code, 2)
                    else:
                        args = flatten.parse_args(argv)
                        self.assertEqual((args.source, args.target), (source, target))

    def test_named_fits_images_and_beam_paths(self):
        for source, directory, expected in (
            ("saved-pupil", "b-pupils", self.clear),
            ("saved-ref", "b-references", self.reference),
        ):
            with self.subTest(source=source):
                self.write_pair(directory, beam=3)
                actual = flatten.load_saved_image(3, source)
                onp.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, onp.dtype(float))
                actual[:] = 0
                onp.testing.assert_array_equal(flatten.load_saved_image(3, source), expected)

    def test_numpy_toggle_selects_single_source_image(self):
        with patch.object(flatten, "USE_FITS", False):
            for source, directory, expected in (
                ("saved-pupil", "b-pupils", self.clear),
                ("saved-ref", "b-references", self.reference),
            ):
                with self.subTest(source=source):
                    path = self.source_path(directory, beam=2, suffix=".npy")
                    onp.save(path, expected.astype(onp.int32))
                    actual = flatten.load_saved_image(2, source)
                    onp.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, onp.dtype(float))

    def test_no_fallback_to_numpy_or_other_beam(self):
        onp.save(self.source_path("b-pupils", suffix=".npy"), self.clear)
        self.write_pair("b-pupils", beam=2)
        with self.assertRaisesRegex(ValueError, "beam1.fits"):
            flatten.load_saved_image(1, "saved-pupil")

    def test_missing_fits_extension_is_rejected(self):
        path = self.source_path("b-references")
        fits.PrimaryHDU(self.clear).writeto(path)
        with self.assertRaisesRegex(ValueError, "PHASE_MASK"):
            flatten.load_saved_image(1, "saved-ref")

    def test_invalid_images_fail_before_hardware_setup(self):
        invalid = (
            onp.ones((16, 16)), onp.ones((2, 32, 32)),
            onp.zeros((32, 32)), -onp.ones((32, 32)),
            onp.full((32, 32), onp.nan), onp.full((32, 32), onp.inf),
        )
        path = self.source_path("b-pupils")
        with patch.object(flatten.zmq, "Context") as connect:
            for data in invalid:
                with self.subTest(shape=data.shape, value=data.flat[0]):
                    primary = fits.PrimaryHDU(data)
                    primary.header["EXTNAME"] = "CLEAR_PUPIL"
                    primary.writeto(path, overwrite=True)
                    with self.assertRaises(SystemExit) as error:
                        flatten.main(["1", "--source", "saved-pupil"])
                    self.assertEqual(error.exception.code, 2)
            connect.assert_not_called()

    def test_non_numeric_and_pickle_arrays_are_rejected(self):
        path = self.source_path("b-pupils", suffix=".npy")
        with patch.object(flatten, "USE_FITS", False):
            for dtype in (str, object, complex, bool):
                with self.subTest(dtype=dtype):
                    onp.save(path, onp.ones((32, 32)).astype(dtype))
                    with self.assertRaises(ValueError):
                        flatten.load_saved_image(1, "saved-pupil")

    def test_missing_input_and_invalid_combination_fail_before_hardware_setup(self):
        with patch.object(flatten.zmq, "Context") as connect:
            for argv in (
                ["1", "--source", "saved-ref"],
                ["1", "--source", "saved-ref", "--target", "stddev"],
            ):
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                    flatten.main(argv)
                self.assertEqual(error.exception.code, 2)
            connect.assert_not_called()

    def test_all_sources_with_mocked_hardware(self):
        self.write_pair("b-pupils")
        self.write_pair("b-references")
        for source, target in (
            ("live", "stddev"), ("live", "model"), ("live", "amp-model"),
            ("saved-pupil", "stddev"), ("saved-pupil", "model"),
            ("saved-pupil", "amp-model"), ("saved-ref", "model"),
            ("saved-ref", "amp-model"),
        ):
            with self.subTest(source=source, target=target), ExitStack() as stack:
                package = ModuleType("asgard_alignment")
                modes = ModuleType("asgard_alignment.DM_modes2")
                camera_module = ModuleType("asgard_alignment.bcam")
                dm_module = ModuleType("asgard_alignment.DM_shm_ctrl")
                package.DM_modes2 = modes
                basis = Mock(num_modes=2)
                basis.linear_combination.return_value = onp.zeros(144)
                basis.coefficients_for.return_value = onp.zeros(2)
                modes.make_hc_act_grid = Mock()
                modes.fourier_basis = Mock(return_value=(basis, None))
                camera = Mock()
                camera.take_stack.side_effect = lambda count: (
                    self.clear[None].copy() if count == 1000
                    else 3 * self.reference[None].copy()
                )
                camera_module.Bcam = Mock(return_value=camera)
                dm = Mock()
                dm_module.dmclass = Mock(return_value=dm)
                stack.enter_context(patch.dict("sys.modules", {
                    "asgard_alignment": package,
                    modes.__name__: modes,
                    camera_module.__name__: camera_module,
                    dm_module.__name__: dm_module,
                }))
                context = stack.enter_context(patch.object(flatten.zmq, "Context"))
                socket = context.return_value.socket.return_value
                socket.recv_string.return_value = "42.0"
                stack.enter_context(patch.object(flatten.time, "sleep"))
                stack.enter_context(patch("builtins.input", return_value="no"))
                subprocess_run = stack.enter_context(patch.object(flatten.subprocess, "run"))
                prepare = stack.enter_context(patch.object(flatten, "prepare_pupil"))
                prepare.return_value = (
                    onp.ones((32, 32)), onp.ones((32, 32)) / 1024, (0, 0), 8,
                )
                generate = stack.enter_context(patch.object(
                    flatten, "generate_target_image", return_value=self.reference.copy()
                ))
                losses = []

                def optimize(fun, x0, args, **kwargs):
                    value = fun(x0, *args)
                    losses.append(value)
                    return SimpleNamespace(x=x0, fun=value)

                stack.enter_context(patch.object(flatten.opt, "minimize", side_effect=optimize))
                flatten.main(["1", "--source", source, "--target", target, "--no-plots"])

                camera.take_dark.assert_called_once_with(256)
                messages = [entry.args[0] for entry in socket.send_string.call_args_list]
                self.assertEqual(messages[:3], [
                    "read BMY1", "moveabs BMY1 500.0", "moveabs BMY1 42.0",
                ])
                self.assertEqual(camera.take_stack.call_args_list.count(call(1000)), int(source == "live"))
                self.assertEqual(len(messages), 7 if source == "live" else 3)
                self.assertEqual(camera.take_stack.call_args_list.count(call(64)), 3)
                if source == "saved-ref":
                    prepare.assert_not_called()
                    generate.assert_not_called()
                else:
                    prepare.assert_called_once()
                    onp.testing.assert_array_equal(prepare.call_args.args[0], self.clear)
                    if target == "stddev":
                        generate.assert_not_called()
                    else:
                        generate.assert_called_once()
                        self.assertEqual(generate.call_args.args[0], target)
                if target != "stddev":
                    onp.testing.assert_allclose(losses, 0, atol=1e-15)
                else:
                    self.assertTrue(onp.isfinite(losses).all())
                onp.testing.assert_array_equal(dm.set_data.call_args.args[0], onp.zeros(144))
                subprocess_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
