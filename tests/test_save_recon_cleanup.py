import argparse
import signal
import unittest
from unittest import mock

from baldr_tt import save_recon


class ReconCleanupTests(unittest.TestCase):
    def setUp(self):
        save_recon._CANCEL.clear()
        self.addCleanup(save_recon._CANCEL.clear)
        self.socket = mock.Mock()

    def test_success_does_not_reset_dm(self):
        with mock.patch.object(save_recon, "get_zmq_socket", return_value=self.socket), mock.patch.object(
            save_recon, "_run_pokes_and_save", return_value=("data", "recon")
        ), mock.patch.object(save_recon, "stop_and_clear_beam") as reset:
            self.assertEqual(save_recon.run_pokes_and_save(2), ("data", "recon"))

        reset.assert_not_called()
        self.socket.s.close.assert_called_once_with(0)

    def test_beam_error_resets_dm_and_preserves_failure(self):
        with mock.patch.object(save_recon, "get_zmq_socket", return_value=self.socket), mock.patch.object(
            save_recon, "_run_pokes_and_save", side_effect=ValueError("bad frame")
        ), mock.patch.object(save_recon, "stop_and_clear_beam") as reset:
            with self.assertRaisesRegex(ValueError, "bad frame"):
                save_recon.run_pokes_and_save(3)

        reset.assert_called_once_with(3)

    def test_sigterm_cancels_and_resets_active_beam(self):
        def interrupted(_beam, _socket):
            save_recon._request_cancel(signal.SIGTERM, None)

        with mock.patch.object(save_recon, "get_zmq_socket", return_value=self.socket), mock.patch.object(
            save_recon, "_run_pokes_and_save", side_effect=interrupted
        ), mock.patch.object(save_recon, "stop_and_clear_beam") as reset:
            with self.assertRaises(save_recon.ReconCancelled):
                save_recon.run_pokes_and_save(1)

        self.assertTrue(save_recon._CANCEL.is_set())
        reset.assert_called_once_with(1)

    def test_cleanup_failure_is_reported_as_failure(self):
        with mock.patch.object(save_recon, "get_zmq_socket", return_value=self.socket), mock.patch.object(
            save_recon, "_run_pokes_and_save", side_effect=ValueError("bad frame")
        ), mock.patch.object(
            save_recon, "stop_and_clear_beam", side_effect=RuntimeError("server unavailable")
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                save_recon.run_pokes_and_save(4)

    def test_multi_beam_failure_exits_unsuccessfully(self):
        args = argparse.Namespace(
            beam=-1, amp=0.04, n_modes=11, settle_sec=1.0, n_iter=1, overwrite=True
        )

        def run_beam(beam):
            if beam == 2:
                raise RuntimeError("beam failed")

        with mock.patch.object(save_recon, "parse_args", return_value=args), mock.patch.object(
            save_recon, "run_pokes_and_save", side_effect=run_beam
        ):
            self.assertEqual(save_recon.main(), 1)

    def test_sigterm_exits_unsuccessfully(self):
        args = argparse.Namespace(
            beam=1, amp=0.04, n_modes=11, settle_sec=1.0, n_iter=1, overwrite=True
        )
        with mock.patch.object(save_recon, "parse_args", return_value=args), mock.patch.object(
            save_recon,
            "run_pokes_and_save",
            side_effect=lambda _beam: save_recon._request_cancel(signal.SIGTERM, None),
        ):
            self.assertEqual(save_recon.main(), 143)


if __name__ == "__main__":
    unittest.main()
