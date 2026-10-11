import json
import unittest
from unittest.mock import Mock, patch

from mcs_client.mcs_client import MCSClient, MCSServer


class MCSClientTests(unittest.TestCase):
    def setUp(self):
        with patch.multiple(
            "mcs_client.mcs_client",
            ZmqReq=Mock(),
            Watchdog=Mock(),
            MCSServer=Mock(),
            HeimdallrAdapter=Mock(),
        ):
            self.client = MCSClient(
                {"HDLR": "inproc://hdlr"}, "inproc://mcs", "inproc://wag"
            )
        self.client.server_z.read_most_recent_msg.return_value = {}
        self.client.gather_hdlr_parameters = Mock(
            return_value=[{"name": "hdlr_locked", "value": 0}]
        )
        self.client._send = Mock(return_value=(False, None))
        self.now = 0.0
        clock = patch(
            "mcs_client.mcs_client.time.monotonic", side_effect=lambda: self.now
        )
        clock.start()
        self.addCleanup(clock.stop)

    def test_requests_are_serviced_between_publications(self):
        self.client.server_z.has_new_data = False
        self.client.server_z.data = {}
        self.client.server_z.fetch.side_effect = [None, None, None, StopIteration]
        self.client.publish_all_to_wag = Mock()
        with patch("mcs_client.mcs_client.time.sleep", side_effect=StopIteration):
            with self.assertRaises(StopIteration):
                self.client.run()
        self.assertEqual(self.client.server_z.fetch.call_count, 4)
        self.client.publish_all_to_wag.assert_called_once_with()

    def test_script_result_is_collected_before_next_status_request(self):
        server = MCSServer.__new__(MCSServer)
        server.z = Mock()
        server.watchdog = Mock()
        server.data = {}
        server.has_new_data = False
        messages = iter([
            "status",
            json.dumps({"origin": "find_fringes", "data": [{"hdlr_complete": True}]}),
            "disk_status",
        ])

        def fetch():
            server.handle_message(next(messages))
            server.has_new_data = True

        server.fetch = fetch
        self.client.server_z = server
        self.client.gather_hdlr_parameters.return_value = []
        self.client._send.return_value = (True, {"reply": {"content": "OK"}})
        with patch("mcs_client.mcs_client.time.sleep", side_effect=StopIteration):
            with self.assertRaises(StopIteration):
                self.client.run()
        self.client._send.assert_called_once()
        self.assertEqual(
            self.client._send.call_args.args[0]["command"]["parameter"],
            [{"name": "hdlr_complete", "value": True}],
        )

    def test_failed_writes_back_off_and_recovery_restores_normal_publication(self):
        for expected_delay in (1.0, 2.0, 4.0, 8.0, 10.0, 10.0):
            with self.assertLogs(level="WARNING"):
                self.client.publish_all_to_wag()
            self.assertEqual(self.client.wag_next_write_at, self.now + expected_delay)
            attempts = self.client._send.call_count
            self.now += expected_delay / 2
            self.client.publish_all_to_wag()
            self.assertEqual(self.client._send.call_count, attempts)
            self.assertEqual(self.client.gather_hdlr_parameters.call_count, attempts)
            self.now = self.client.wag_next_write_at

        self.client._send.return_value = (True, {"reply": {"content": "OK"}})
        with self.assertLogs(level="INFO") as logs:
            self.client.publish_all_to_wag()
        self.assertTrue(any("recovered" in message for message in logs.output))
        self.assertEqual(self.client.wag_write_backoff_s, 0.0)
        self.assertEqual(self.client.wag_next_write_at, 0.0)
        attempts = self.client._send.call_count
        self.client.publish_all_to_wag()
        self.assertEqual(self.client._send.call_count, attempts + 1)

    def test_script_values_survive_outage_with_fresh_telemetry_on_retry(self):
        self.client.server_z.read_most_recent_msg.side_effect = [
            {"beam": 0, "data": [{"offset": 1}]},
            {"beam": 0, "data": [{"offset": 2}]},
            {"beam": 1, "data": [{"offset": 3}]},
            {},
        ]
        with self.assertLogs(level="WARNING"):
            self.client.publish_all_to_wag()
        self.client.publish_all_to_wag()
        self.client.publish_all_to_wag()
        self.client._send.assert_called_once()

        self.now = self.client.wag_next_write_at
        self.client.gather_hdlr_parameters.return_value = [
            {"name": "hdlr_locked", "value": 1}
        ]
        self.client._send.return_value = (True, {"reply": {"content": "OK"}})
        self.client.publish_all_to_wag()
        self.assertEqual(
            self.client._send.call_args.args[0]["command"]["parameter"],
            [
                {"name": "hdlr_locked", "value": 1},
                {"name": "offset", "value": [2], "range": "(0:0)"},
                {"name": "offset", "value": [3], "range": "(1:1)"},
            ],
        )
        self.assertEqual(self.client._pending_script_params, {})

    def test_overlapping_script_ranges_keep_latest_write_order(self):
        full = {"name": "offset", "value": [4, 4, 4, 4], "range": "(0:3)"}
        beam = {"name": "offset", "value": [2], "range": "(0:0)"}
        self.client.gather_script_parameters = Mock(side_effect=[
            [{**full, "value": [1, 1, 1, 1]}], [beam], [full], [],
        ])
        with self.assertLogs(level="WARNING"):
            self.client.publish_all_to_wag()
        self.client.publish_all_to_wag()
        self.client.publish_all_to_wag()
        self.now = self.client.wag_next_write_at
        self.client._send.return_value = (True, {"reply": {"content": "OK"}})
        self.client.publish_all_to_wag()
        self.assertEqual(
            self.client._send.call_args.args[0]["command"]["parameter"][1:],
            [beam, full],
        )

    def test_invalid_wag_reply_reconnects_without_crashing(self):
        good_reply = {"reply": {"content": "OK"}}
        self.client.req_z.s.closed = False
        for invalid in (None, [], {"unexpected": 1}, {"reply": []},
                        {"reply": {}}, {"reply": {"content": None}}):
            with self.subTest(reply=invalid):
                self.client.req_z.reconnect.reset_mock()
                self.client.req_z.send_payload.side_effect = [invalid, good_reply]
                with patch("mcs_client.mcs_client.time.sleep"), self.assertLogs(
                    level="WARNING"
                ):
                    result = MCSClient._send(self.client, {})
                self.assertEqual(result, (True, good_reply))
                self.client.req_z.reconnect.assert_called_once_with()

    def test_explicit_wag_error_is_not_accepted_as_success(self):
        reply = {"reply": {"content": "ERROR"}}
        self.client.req_z.s.closed = False
        self.client.req_z.send_payload.return_value = reply
        self.assertEqual(MCSClient._send(self.client, {}), (False, reply))
        self.client.req_z.send_payload.assert_called_once_with({})


if __name__ == "__main__":
    unittest.main()
