import zmq
import sys

REQUEST_TIMEOUT_MS = 1000  # Time to wait for a reply before retrying
REQUEST_RETRIES = 3  # Number of times to retry before giving up


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

    def send_and_recv(self, message, retries=REQUEST_RETRIES):
        for attempt in range(1, retries + 1):
            self.socket.send_string(message)
            if self.socket.poll(self.timeout_ms, zmq.POLLIN):
                return self.socket.recv_string().strip()

            print(
                f"WARN: Timeout waiting for reply to '{message}' "
                f"(attempt {attempt}/{retries})."
            )
            self.socket.close()
            self.socket = self._create_socket()

        print(
            f"ERROR: No reply from ADC server after {retries} attempts for '{message}'."
        )
        sys.exit(1)

    def close(self):
        self.socket.close()
