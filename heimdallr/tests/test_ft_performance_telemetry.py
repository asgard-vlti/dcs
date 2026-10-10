import tempfile
import time
import unittest
from pathlib import Path

from heimdallr.save_ft_performance import FTPerformanceLogger, wait_for_telemetry


def row(seq, mode=1):
    return {
        "seq": seq,
        "cnt": seq % 10000,
        "servo_mode": mode,
        "time_ns": 1_600_000_000_000_000_000 + 250_000 * seq,
        "gd_snr": [1.0] * 6,
        "pd_snr": [2.0] * 6,
        "gd_bl": [3.0] * 6,
        "pd_tel": [4.0] * 4,
        "gd_tel": [5.0] * 4,
        "dm_piston": [6.0] * 4,
    }


def reply(stream_id, latest_seq, rows=(), *, reset=False, overrun=False, dropped=0):
    return {
        "stream_id": stream_id,
        "oldest_seq": rows[0]["seq"] if rows else latest_seq,
        "latest_seq": latest_seq,
        "reset": reset,
        "overrun": overrun,
        "dropped_total": dropped,
        "rows": list(rows),
    }


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def send_payload(self, payload, **kwargs):
        self.requests.append(payload)
        return next(self.responses)


def make_logger(directory, responses, initial=None):
    client = FakeClient(responses)
    path = directory / "ft.log"
    logger = FTPerformanceLogger(client, initial or reply(1, 0), path)
    return logger, client, path


class FTPerformanceTelemetryTest(unittest.TestCase):
    def test_retry_keeps_cursor_and_writes_batch_once(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = [row(1), row(2)]
            client_reply = reply(1, 2, rows)
            logger, client, path = make_logger(
                Path(temp), [None, client_reply, reply(1, 2)]
            )
            try:
                with self.assertRaises(ConnectionError):
                    logger.poll_once()
                self.assertEqual(logger.last_seq, 0)
                self.assertFalse(logger.poll_once())
                self.assertFalse(logger.poll_once())
            finally:
                logger.close()
            self.assertEqual(
                client.requests,
                [
                    "ft_telemetry 1,0,128",
                    "ft_telemetry 1,0,128",
                    "ft_telemetry 1,2,128",
                ],
            )
            lines = [
                line
                for line in path.read_text().splitlines()
                if not line.startswith("#")
            ]
            self.assertEqual(len(lines), 2)
            self.assertEqual([line.split()[-2] for line in lines], ["1", "2"])
            self.assertEqual(len(lines[0].split()), 34)
            self.assertEqual(lines[0].split()[0], "1600000000.0003")

    def test_loss_restart_and_mode_from_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            logger, _, path = make_logger(
                Path(temp),
                [
                    reply(1, 6, [row(5), row(6, mode=4)], overrun=True),
                    reply(2, 1, [row(1)], reset=True),
                    reply(2, 2, [row(2)], dropped=1),
                    reply(2, 3, [row(3)], dropped=1),
                ],
            )
            try:
                logger.poll_once()
                self.assertEqual(logger.servo_mode, 4)
                logger.poll_once()
                self.assertEqual(logger.servo_mode, 1)
                logger.poll_once()
                logger.poll_once()
            finally:
                logger.close()
            content = path.read_text()
            self.assertEqual(content.count("# telemetry_restart"), 1)
            self.assertEqual(content.count("# telemetry_loss missing=4"), 1)
            self.assertEqual(content.count("# telemetry_loss producer_dropped=1"), 1)

    def test_wrap_is_contiguous(self):
        with tempfile.TemporaryDirectory() as temp:
            logger, _, path = make_logger(
                Path(temp),
                [reply(1, 1, [row(0xFFFFFFFE), row(0xFFFFFFFF), row(0), row(1)])],
                reply(1, 0xFFFFFFFD),
            )
            try:
                logger.poll_once()
            finally:
                logger.close()
            content = path.read_text()
            self.assertNotIn("# telemetry_loss", content)
            self.assertEqual(
                [
                    int(line.split()[-2])
                    for line in content.splitlines()
                    if not line.startswith("#")
                ],
                [0xFFFFFFFE, 0xFFFFFFFF, 0, 1],
            )

    def test_old_server_is_rejected(self):
        client = FakeClient([None, {"cnt": 4}])
        with self.assertRaisesRegex(RuntimeError, "deploy the updated server first"):
            wait_for_telemetry(client)
        self.assertEqual(client.requests, ["ft_telemetry 0,0,0", "status"])

    def test_restart_reports_unavailable_new_stream_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            logger, _, path = make_logger(
                Path(temp),
                [reply(2, 1500, [row(1500)], reset=True, dropped=3)],
            )
            try:
                logger.poll_once()
            finally:
                logger.close()
            content = path.read_text()
            self.assertEqual(content.count("# telemetry_restart"), 1)
            self.assertEqual(content.count("# telemetry_loss since_restart=1499"), 1)
            self.assertNotIn("# telemetry_loss producer_dropped", content)

    def test_drains_4000_rows_in_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            responses = [
                reply(1, 4000, [row(seq) for seq in range(start, min(start + 128, 4001))])
                for start in range(1, 4001, 128)
            ]
            logger, client, path = make_logger(Path(temp), responses)
            try:
                started = time.perf_counter()
                for index in range(len(responses)):
                    self.assertEqual(logger.poll_once(), index < len(responses) - 1)
                elapsed = time.perf_counter() - started
            finally:
                logger.close()
            self.assertEqual(len(client.requests), 32)
            self.assertEqual(logger.last_seq, 4000)
            content = path.read_text()
            self.assertNotIn("# telemetry_loss", content)
            self.assertEqual(
                sum(not line.startswith("#") for line in content.splitlines()), 4000
            )
            print(f"FT logger drained 4000 rows in {elapsed:.3f} s")


if __name__ == "__main__":
    unittest.main()
