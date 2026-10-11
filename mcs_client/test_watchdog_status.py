import unittest
from unittest.mock import Mock, patch

import zmq

from mcs_client.mcs_client import Watchdog


class WatchdogStatusTests(unittest.TestCase):
    def setUp(self):
        self.watchdog = Watchdog.__new__(Watchdog)
        self.watchdog.watchdog_fast_retries = 1
        self.watchdog.watchdog_fast_timeout_ms = 80
        self.zmq_obj = Mock()
        self.zmq_obj.s = Mock()

    def test_busy_server_keeps_process_running_when_tcp_listener_is_open(self):
        for name, endpoint, port in (
            ("MDS", "tcp://mimir:5555", 5555),
            ("BTT1", "tcp://mimir:6671", 6671),
        ):
            with self.subTest(name=name):
                self.zmq_obj.endpoint = endpoint
                with patch("mcs_client.mcs_client.zmq.Poller") as poller, patch(
                    "mcs_client.mcs_client.check_port", return_value="open"
                ) as check:
                    poller.return_value.poll.return_value = []
                    status = self.watchdog._watchdog_lazy_pirate_status(
                        name, self.zmq_obj
                    )

                self.assertEqual(status, ("running", "no-reply", "no-reply"))
                check.assert_called_once_with(port, host="mimir")

    def test_missing_listener_reports_closed(self):
        self.zmq_obj.endpoint = "tcp://mimir:5555"
        with patch("mcs_client.mcs_client.zmq.Poller") as poller, patch(
            "mcs_client.mcs_client.check_port", return_value="closed"
        ):
            poller.return_value.poll.return_value = []
            status = self.watchdog._watchdog_lazy_pirate_status("MDS", self.zmq_obj)

        self.assertEqual(status, ("closed", "closed", "no-reply"))

    def test_status_reply_reports_running_without_tcp_probe(self):
        self.zmq_obj.endpoint = "tcp://mimir:5555"
        self.zmq_obj.s.recv_string.return_value = "ready"
        with patch("mcs_client.mcs_client.zmq.Poller") as poller, patch(
            "mcs_client.mcs_client.check_port"
        ) as check:
            poller.return_value.poll.return_value = {
                self.zmq_obj.s: zmq.POLLIN
            }
            status = self.watchdog._watchdog_lazy_pirate_status("MDS", self.zmq_obj)

        self.assertEqual(status, ("running", "open", "ready"))
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
