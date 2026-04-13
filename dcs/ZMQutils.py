import json
import zmq
from typing import Any, Dict, Optional
import logging
import base64

class ZmqReq:
    """
    An adapter for a ZMQ REQ socket (client).
    """

    def __init__(self, endpoint: str, timeout_ms: int = 1500):
        self.ctx = zmq.Context.instance()
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self._had_error = False
        self.s = self.ctx.socket(zmq.REQ)
        self.s.RCVTIMEO = timeout_ms
        self.s.SNDTIMEO = timeout_ms
        self.s.connect(endpoint)

    def _reconnect_socket(self) -> None:
        try:
            self.s.disconnect(self.endpoint)
        except zmq.ZMQError:
            # Socket may already be disconnected or in a transitional state.
            pass

        self.s.close(0)
        self.s = self.ctx.socket(zmq.REQ)
        self.s.RCVTIMEO = self.timeout_ms
        self.s.SNDTIMEO = self.timeout_ms
        self.s.connect(self.endpoint)
        self._had_error = False

    def _send_string(self, payload: str) -> bool:
        if self._had_error:
            self._reconnect_socket()

        try:
            self.s.send_string(payload)
            return True
        except zmq.ZMQError as e:
            logging.error(f"ZMQ send error occurred: {e}")
            self._had_error = True
            return False

    def send_payload(
        self, payload: Dict[str, Any], is_str=False, decode_ascii=True, image=False
    ) -> Optional[Dict[str, Any]]:
        """
        is_str: if True, payload is a string and will be sent as-is
        decode_ascii: if True, decode the response as ascii and strip the last character (used False for Cpp interfaces)
        """
        if not is_str:
            if not self._send_string(json.dumps(payload, sort_keys=True)):
                return None
        else:
            if not self._send_string(payload):
                return None

        try:
            if decode_ascii:
                res = self.s.recv().decode("ascii")[:-1]
            else:
                res = self.s.recv_string()
                
            jres = json.loads(res)
            self._had_error = False
            if image:
                return image_from_message(jres)
            return json.loads(res)
        except zmq.error.Again as e:
            logging.error(f"ZMQ error occurred: {e}")
            self._had_error = True
            return None
        except zmq.ZMQError as e:
            logging.error(f"ZMQ error occurred: {e}")
            self._had_error = True
            return None
        except json.decoder.JSONDecodeError:
            logging.error(f"JSON decode error occurred")
            return None
        
def image_from_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a message containing an image in base64 to a numpy array.
    """
    import numpy as np
    import base64

    if msg['type'] == 'complex':
        dtype = np.complex64
    elif msg['type'] == 'double':
        dtype = np.float64
    elif msg['type'] == 'float':
        dtype = np.float32
    elif msg['type'] == 'int32':
        dtype = np.int32
    elif msg['type'] == 'int64':
        dtype = np.int64
    elif msg['type'] == 'uint8':
        dtype = np.uint8
    elif msg['type'] == 'uint16':
        dtype = np.uint16
    elif msg['type'] == 'uint32':
        dtype = np.uint32
    elif msg['type'] == 'uint64':
        dtype = np.uint64
    elif msg['type'] == 'int8':
        dtype = np.int8
    elif msg['type'] == 'int16':
        dtype = np.int16
    elif msg['type'] == 'float16':
        dtype = np.float16
    else:
        raise ValueError(f"Unknown type {msg['dtype']}")
    
    dat = np.frombuffer(base64.b64decode(msg['message']), dtype=dtype)
    dat = dat.reshape((msg['szx'], msg['szy']))
    msg['image_data'] = dat
    return msg
