import zmq
import sys
import logging

REQUEST_TIMEOUT_MS = 1000  # Time to wait for a reply before retrying
REQUEST_RETRIES = 3  # Number of times to retry before giving up


logger = logging.getLogger(__name__)


class ZmqLazyPirateClient:
    def __init__(self, context, endpoint, timeout_ms=REQUEST_TIMEOUT_MS):
        self.context = context
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.socket = self._create_socket()

    def _create_socket(self):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        return socket

    def send_and_recv(self, message, retries=REQUEST_RETRIES, exit_on_failure=True):
        for attempt in range(1, retries + 1):
            self.socket.send_string(message)
            if self.socket.poll(self.timeout_ms, zmq.POLLIN):
                return self.socket.recv_string().strip()

            logger.warning(
                "Timeout waiting for reply to '%s' (attempt %s/%s).",
                message,
                attempt,
                retries,
            )
            self.socket.close()
            self.socket = self._create_socket()

        logger.error(
            "No reply from ADC server after %s attempts for '%s'.",
            retries,
            message,
        )
        if exit_on_failure:
            sys.exit(1)
        else:
            return None

    def close(self):
        self.socket.close()
