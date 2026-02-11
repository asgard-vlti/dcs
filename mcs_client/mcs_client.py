# baldr_wag_client.py
# TODO: Sometimes, the database parameters (i.e. values in #sym:HeimdallrStatus  and #sym:BaldrTscopeStatus ) will not change with every call. Modify the classes to be aware of their previous state, and only publish changes to database parameters

import json, time, socket
import logging
import os
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, List, Optional, Tuple
import zmq
from datetime import datetime, timezone

# baldr_wag_client.py
import json, time, socket
import logging
import os
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, List, Optional, Tuple
import zmq
from datetime import datetime, timezone

# Following protocol described in
# Top-Level Control Software
# User and Maintenance Manual
# sec 8.7.2

# example (to be discussed with team) of baldr_mcs_client which
# is a lightweight Python bridge that polls Baldr/Heimdallr or other
# ZMQ status and reads/writes the corresponding shared parameters
# on WAGs Module Communication Server (TCP 7020), keeping OLDB in
# sync for operations and GUIs.

# Sockets:
# cam_server      6667
# DM_server       6666
# hdlr            6660
# hdlr_align      6661
# baldr           6662


class ZmqReq:
    """
    An adapter for a ZMQ REQ socket (client).
    """

    def __init__(self, endpoint: str, timeout_ms: int = 1500):
        self.ctx = zmq.Context.instance()
        self.s = self.ctx.socket(zmq.REQ)
        self.s.RCVTIMEO = timeout_ms
        self.s.SNDTIMEO = timeout_ms
        self.s.connect(endpoint)

    def send_payload(
        self, payload: Dict[str, Any], is_str=False, decode_ascii=True
    ) -> Optional[Dict[str, Any]]:
        if not is_str:
            self.s.send_string(json.dumps(payload, sort_keys=True))
        else:
            self.s.send_string(payload)

        try:

            if decode_ascii:
                res = self.s.recv().decode("ascii")[:-1]
            else:
                res = self.s.recv_string()

            return json.loads(res)
        except zmq.error.Again:
            return None
        except json.decoder.JSONDecodeError:
            return None


class ZmqRep:
    """
    An adapter for a ZMQ REP socket (server).
    """

    def __init__(self, endpoint: str):
        self.ctx = zmq.Context.instance()
        self.s = self.ctx.socket(zmq.REP)
        self.s.bind(endpoint)

    def recv_payload(self) -> Optional[Dict[str, Any]]:
        try:
            msg = self.s.recv_string()
            return json.loads(msg)
        except zmq.error.Again:
            return None

    def send_payload(self, payload: Dict[str, Any]) -> bool:
        try:
            self.s.send_string(json.dumps(payload))
            logging.info(f"Sent payload: {payload}")
            return True
        except zmq.error.Again:
            return False


# ---------------- MCS client ----------------
class MCSClient:

    @staticmethod
    def _is_zmq_socket_open(sock):
        """
        Actively checks if the ZMQ socket is open and the remote endpoint is alive.
        For REQ sockets, sends a 'status' or 'ping' and expects a reply.
        Returns True only if a valid response is received.
        """

        # Check if socket is not closed
        if not hasattr(sock, "closed") or sock.closed:
            return False
        # Only works for REQ sockets (client side)
        try:
            if sock.type == zmq.REQ:
                poller = zmq.Poller()
                poller.register(sock, zmq.POLLIN)
                # Try to send a test message (non-blocking, short timeout)
                try:
                    # Use a unique message unlikely to interfere
                    sock.send_string("__ping__", zmq.NOBLOCK)
                except zmq.Again:
                    # Socket not ready to send
                    return False
                # Wait for a reply (short timeout)
                socks = dict(poller.poll(300))
                if sock in socks and socks[sock] == zmq.POLLIN:
                    try:
                        _ = sock.recv(zmq.NOBLOCK)
                        return True
                    except Exception:
                        return False
                else:
                    return False
            # For REP sockets (server side), we can't actively check remote
            # Just check if not closed
            elif sock.type == zmq.REP:
                return True
        except Exception:
            return False
        return False

    def __init__(
        self,
        dcs_endpoints: dict,
        script_endpoint: str,
        publish_endpoint: str,
        sleep_time: float = 1.0,
        script_only: bool = False,
    ):
        self.publish_z = ZmqReq(publish_endpoint)
        logging.info(f"REQ publish set up on {publish_endpoint}")

        self.script_z = ScriptAdapter(script_endpoint)
        logging.info(f"ScriptAdapter(REP) set up on {script_endpoint}")

        self.dcs_endpoints = dcs_endpoints

        self.dcs_adapters = {}
        for dcs_name, endpoint in dcs_endpoints.items():
            if dcs_name.startswith("BLD"):
                self.dcs_adapters[dcs_name] = BaldrAdapter(endpoint)
            elif dcs_name == "HDLR":
                self.dcs_adapters[dcs_name] = HeimdallrAdapter(endpoint)
            else:
                logging.warning(f"Unknown DCS adapter name '{dcs_name}', skipping.")
                continue

        self.requester = "mimir"

        self.sleep_time = sleep_time
        self.script_only = script_only

        self.t_last_server_check = time.time()

    def _send(self, body: Dict[str, Any]) -> Tuple[bool, str]:
        rep = self.publish_z.send_payload(body)
        print(rep)
        if not rep or "reply" not in rep:
            return False, "no-reply"
        content = rep["reply"].get("content", "ERROR")
        return (content == "OK" or content != "ERROR"), str(content)

    def run(self):
        """
        Check if any new data has arrived from the scripts, and
        also poll the cpp databases for new data. Publish all at once.
        """
        while True:
            self.publish_all_to_wag()
            time.sleep(self.sleep_time)

    def gather_baldr_parameters(self):
        """Gather Baldr parameters for all beams as a list of dicts. Only query if connection is open."""
        data = []
        for beam_idx in range(1, 5):
            adapter = self.dcs_adapters.get(f"BLD{beam_idx}")
            if not adapter or not hasattr(adapter, "z") or not hasattr(adapter.z, "s"):
                logging.warning(
                    f"Baldr adapter for beam {beam_idx} not available or not connected."
                )
                return []
            sock = adapter.z.s
            if not self._is_zmq_socket_open(sock):
                logging.warning(
                    f"Baldr ZMQ socket for beam {beam_idx} is not open or not connected."
                )
                return []
            st = adapter.fetch()
            if not st:
                logging.warning(f"no Baldr status for beam {beam_idx}")
                return []
            data.append(st)
        param_list = []
        for field in asdict(data[0]).keys():
            values = [getattr(d, field) for d in data]
            param_list.append(
                {
                    "name": f"bld_{field}",
                    "value": values,
                    "range": "(0:3)",
                }
            )
        return param_list

    def gather_hdlr_parameters(self):
        """Gather HDLR parameters as a list of dicts. Only query if connection is open."""
        adapter = self.dcs_adapters.get("HDLR")
        if not adapter or not hasattr(adapter, "z") or not hasattr(adapter.z, "s"):
            logging.warning("Heimdallr adapter not available or not connected.")
            if time.time() - self.t_last_server_check > 10:
                self.t_last_server_check = time.time()
                self.dcs_adapters["HDLR"] = HeimdallrAdapter(self.dcs_endpoints["HDLR"])
            return []
        sock = adapter.z.s
        if not self._is_zmq_socket_open(sock):
            logging.warning("Heimdallr ZMQ socket is not open or not connected.")
            if time.time() - self.t_last_server_check > 10:
                self.t_last_server_check = time.time()
                self.dcs_adapters["HDLR"] = HeimdallrAdapter(self.dcs_endpoints["HDLR"])
            return []
        st = adapter.fetch()
        if st is None:
            return []
        Hdlr_parameters = [f.name for f in fields(HeimdallrStatus)]
        param_list = []
        for param in Hdlr_parameters:
            if param in ["locked"]:
                param_list.append(
                    {
                        "name": f"hdlr_{param}",
                        "value": getattr(st, param),
                    }
                )
            else:
                values = getattr(st, param)  # is already a list
                param_list.append(
                    {
                        "name": f"hdlr_{param}",
                        "value": values,
                        "range": f"(0:{len(values)-1})",
                    }
                )
        return param_list

    def gather_script_parameters(self):
        """Gather script parameters if new data is available, as a list of dicts. Only query if connection is open."""
        self.script_z.fetch()
        msg = self.script_z.read_most_recent_msg()
        if not msg:
            return None
        if "beam" not in msg:
            data = msg["data"]
            for i, item in enumerate(data):
                if not isinstance(item, dict) or len(item) != 1:
                    logging.warning(f"ignoring malformed script data item: {item}")
                    continue
                key = list(item.keys())[0]
                value = item[key]
                data[i] = {"name": key, "value": value}
            for item in data:
                if isinstance(item.get("value"), (list, tuple)):
                    item["range"] = "(0:3)"
        else:
            beam_no = msg["beam"]
            data = msg["data"]
            for i, item in enumerate(data):
                if not isinstance(item, dict) or len(item) != 1:
                    logging.warning(f"ignoring malformed script data item: {item}")
                    continue
                key = list(item.keys())[0]
                value = item[key]
                data[i] = {
                    "name": key,
                    "value": [value],
                    "range": f"({beam_no}:{beam_no})",
                }
        return data

    def publish_all_to_wag(self):
        """Publish all parameters (baldr, hdlr, script) in a single message."""
        all_params = []

        if not self.script_only:
            # baldr_params = self.gather_baldr_parameters()
            # if baldr_params:
            #     all_params.extend(baldr_params)
            hdlr_params = self.gather_hdlr_parameters()
            if hdlr_params:
                all_params.extend(hdlr_params)
        script_params = self.gather_script_parameters()
        print("script_params", script_params)
        if script_params:
            all_params.extend(script_params)
        if not all_params:
            logging.info("No msgs found, waiting...")
            return
        body = self.ESO_format(all_params)
        # logging.info(f"pushing: {str(body)[:20]}]...")
        logging.info(f"pushing: {str(body)}...")
        try:
            ok, msg = self._send(body)
            if not ok:
                logging.warning(f"failed to write combined data to wag: {msg}")
        except zmq.error.ZMQError as e:
            logging.error(f"ZMQ error to wag: {e}")

    def publish_bld_databases_to_wag(self):
        adapter_names = [f"BLD{idx}" for idx in range(1, 5)]
        for adapter in adapter_names:
            self.dcs_adapters[adapter].fetch()

        # BLD: for each BaldrTscopeStatus, need to change to a list of values
        # for each parameter (per adapter)
        bld_parameters = fields(BaldrTscopeStatus)

        body = self.ESO_format([])

        # append to body["parameter"]
        for param in bld_parameters:
            values = [self.dcs_adapters[x][param] for x in adapter_names]
            prop = {}
            prop["name"] = f"bld_{param}"
            prop["range"] = "(0:3)"
            prop["value"] = values
            body["parameter"].append(prop)

        ok, msg = self._send(body)

        if not ok:
            logging.warning(f"failed to write script data to wag: {msg}")

    # def publish_hdlr_databases_to_wag(self):
    #     self.dcs_adapters["HDLR"].fetch()

    #     Hdlr_parameters = fields(HeimdallrStatus)

    #     body = self.ESO_format([])

    #     # append to body["parameter"]
    #     for param in Hdlr_parameters:
    #         values = self.dcs_adapters["HDLR"][param]  # is already a list
    #         prop = {}
    #         prop["name"] = f"hdlr_{param}"
    #         prop["range"] = "(0:3)"
    #         prop["value"] = values
    #         body["parameter"].append(prop)

    #     ok, msg = self._send(body)
    #     if not ok:
    #         logging.warning(f"failed to write script data to wag: {msg}")

    def publish_script_data(self):
        if self.script_z.has_new_data:
            msg = self.script_z.read_most_recent_msg()
        else:
            return
        # // option 1: all 4 beams updated, formatting done by MCS
        # {
        #     "origin": "s_h-autoalign",
        #     "data": [
        #         {"hdlr_x_offset": x_offsets}, // each value is a list if needed
        #         {"hdlr_y_offset": y_offsets},
        #         {"hdlr_complete": True},
        #     ],
        # }

        # // option 2: single beam updated, formatting done by MCS
        # {
        #     "origin": "s_h-autoalign",
        #     "beam" : beam_no, // if this keyword exists MCS knows it is this case
        #     "data": [
        #         {"hdlr_x_offset": x_offset}, // each value is a single value
        #         {"hdlr_y_offset": y_offset}, // constraint: only params with "unique per telescope" can be sent
        #     ],
        # }

        # final form that is sent to wag is (in the data section):
        # // case 1
        # [
        #     {
        #         "name": "hdlr_x_offset",
        #         "value": x_offsets,
        #         "range": "(0:3)"
        #     }, ... // same for all other params
        # ]

        # // case 2
        # [
        #     {
        #         "name": "hdlr_x_offset",
        #         "value": [x_offset],
        #         "range": "(beam_no:beam_no)"
        #     }, ... // same for all other params
        # ]

        # check if "beam" keyword exists, if so it is a single beam update
        # otherwise it is all beams
        if "beam" not in msg:
            # case 1
            # case 2
            data = msg["data"]
            for i, item in enumerate(data):
                if not isinstance(item, dict) or len(item) != 1:
                    logging.warning(f"ignoring malformed script data item: {item}")
                    continue
                key = list(item.keys())[0]
                value = item[key]
                data[i] = {"name": key, "value": value}

            # for any lists in data, need to add the "range" field
            for item in data:
                if isinstance(item.get("value"), (list, tuple)):
                    item["range"] = "(0:3)"

        else:
            beam_no = msg["beam"]

            data = msg["data"]

            for i, item in enumerate(data):
                if not isinstance(item, dict) or len(item) != 1:
                    logging.warning(f"ignoring malformed script data item: {item}")
                    continue
                key = list(item.keys())[0]
                value = item[key]
                data[i] = {
                    "name": key,
                    "value": [value],
                    "range": f"({beam_no}:{beam_no})",
                }

        # # write all fields to MCS in a single message
        body = self.ESO_format(data)
        ok, msg = self._send(body)

        if not ok:
            logging.warning(f"failed to write script data to wag: {msg}")

    def ESO_format(self, content):
        return {
            "command": {
                "name": "write",
                "time": self.ts(),
                "parameter": content,
                "requester": self.requester,
            }
        }

    def publish_baldr_to_wag(self):
        """
        example - publish all of baldr.

        Need to fetch for all beams, then group each paramaeter into a list of size 4
        then send that in one packet to wag
        """
        data = []

        for beam_idx in range(1, 5):
            st = self.dcs_adapters[f"BLD{beam_idx}"].fetch()
            if not st:
                logging.warning(f"no Baldr status for beam {beam_idx}")
                return
            data.append(st)

        # write all fields to MCS in a single message
        body = self.ESO_format([])

        # populate the parameter list with all fields from the dataclass,
        # each field will be a list of 4 values (one per beam)
        param_list = []
        for field in asdict(data[0]).keys():
            values = [getattr(d, field) for d in data]
            param_list.append(
                {
                    "name": f"bld_{field}",
                    "value": values,
                    "requester": self.requester,
                }
            )

        # update body
        body["command"]["parameter"] = param_list

        ok, msg = self._send(body)
        if not ok:
            logging.warning(f"failed to write baldr1_status to wag: {msg}")

    @staticmethod
    def ts():
        current_utc_time = datetime.now(timezone.utc)
        # Format the UTC time
        return current_utc_time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------- Server adapters ----------------
@dataclass
class BaldrTscopeStatus:
    """
    data class for all "unique per telescope" status fields
    """

    TT_state: int
    HO_state: int
    mode: str
    phasemask: str
    frequency: float
    configured: int
    ctrl_type: str
    # complete: bool
    config_file: str
    inj_enabled: int
    auto_loop: int
    close_on_snr: float
    open_on_snr: float
    close_on_strehl: float
    open_on_strehl: float
    TT_offsets: int
    # x_pup_offset: float
    # y_pup_offset: float


class CppServerAdapter:
    def __init__(self, endpoint: str):
        self.z = ZmqReq(endpoint)

    def fetch(self) -> Optional[Any]:
        raise NotImplementedError


class BaldrAdapter(CppServerAdapter):
    """
    Talks to Baldr ZMQ server ("tcp://host:6662 ... 6665") and returns a BaldrStatus.
    """

    def __init__(self, endpoint):
        super().__init__(endpoint)

        self.cur_status = None

    def fetch(self) -> Optional[BaldrTscopeStatus]:
        """
        Queries Baldr Cpp servers. Expected return of the form:
        {
            "ok": True,
            "status": {
                "TT_state": int,
                ... etc
            }
        }
        """
        # Example: ask a 'status' command.. this needs to be defined in the baldr or heim commander functs
        rep = self.z.send_payload("status", is_str=True, decode_ascii=False)

        logging.info(f"BaldrAdapter.fetch() got reply (type {type(rep)}): {rep}")
        if not rep:
            return None

        # ---- Map server reply -> BaldrStatus  ----
        # Below assumes the server returns a dict with keys matching your fields.
        try:
            st = rep
            # Use dataclass fields and kwargs to construct BaldrTscopeStatus
            field_names = [f.name for f in fields(BaldrTscopeStatus)]
            kwargs = {}
            for name in field_names:
                if name in st:
                    kwargs[name] = st[name]
            return BaldrTscopeStatus(**kwargs)
        except KeyError as e:
            logging.warning(f"KeyError in BaldrAdapter.fetch(): {e}")
            return None


@dataclass
class HeimdallrStatus:
    gd_snr: list[float]
    pd_snr: list[float]
    v2_K1: list[float]
    v2_K2: list[float]
    dl_offload: list[float]
    locked: int


class HeimdallrAdapter(CppServerAdapter):
    """
    Talks to Heimdallr ZMQ server ("tcp://host:6660") and returns a HeimdallrStatus.
    """

    def __init__(self, endpoint):
        super().__init__(endpoint)

    def fetch(self) -> Optional[HeimdallrStatus]:
        rep = self.z.send_payload("status", is_str=True, decode_ascii=False)
        if not rep:
            return None

        try:
            st = rep
            # Use dataclass fields and kwargs to construct HeimdallrStatus
            field_names = [f.name for f in fields(HeimdallrStatus)]
            kwargs = {}
            for name in field_names:
                if name in st:
                    if name == "locked":
                        print(name, st[name])
                        kwargs[name] = int(st[name])
                    else:
                        kwargs[name] = st[name]
            return HeimdallrStatus(**kwargs)
        except KeyError:
            logging.warning("KeyError in HeimdallrAdapter.fetch()")
            return None


class ScriptAdapter:
    """
    A dedicated channel for talking to DCS command scripts that have been run
    This differs from the C++ servers in that it has be the server side of a ZMQ REQ/REP pair
    and the scripts are the clients. When the data is dumped by the client, this class
    just returns "ack" to the script and the data is saved in MCS.

    The expectation for the data format is a list of dicts like:
    [
        {"name": "param1", "value": val1},
        {"name": "param2", "value": val2},
        ...
    ]
    noting that the MCS will append the requester field automatically.
    """

    def __init__(self, endpoint: str):
        self.z = ZmqRep(endpoint)

        self.poller = zmq.Poller()
        self.poller.register(self.z.s, zmq.POLLIN)

        self.data = {}
        self.has_new_data = False

    def read_most_recent_msg(self):
        if self.has_new_data == False:
            return {}

        self.has_new_data = False

        return self.data

    def fetch(self) -> Optional[Dict[str, Any]]:
        socks = dict(self.poller.poll(10))
        inputready = []
        if self.z.s in socks and socks[self.z.s] == zmq.POLLIN:
            inputready.append(self.z.s)
        for s in inputready:  # loop through our array of sockets/inputs
            msg = self.socket_funct(s)
            logging.info(f"Received message: {msg}")
            self.handle_message(msg)
            self.has_new_data = True

    def socket_funct(self, s):
        try:
            message = s.recv_string()
            return message
        except zmq.ZMQError as e:
            # logging.error(f"ZMQ Error: {e}")
            return -1

    def handle_message(self, msg):

        msg = json.loads(msg)
        # Acknowledge receipt
        if not msg:
            return None
        self.z.send_payload({"ok": True})

        if msg.get("origin") == "s_h-autoalign":
            self.data = msg
        elif msg.get("origin") == "s_h-shutter":
            self.data = msg
        elif msg.get("origin") == "s_bld_pup_autoalign_sky":
            # Save the data for later processing
            self.data = msg


# ---------------- Main publish loop ----------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the MDS server.")
    parser.add_argument(
        "--log-location",
        type=str,
        default="~/logs/mcs/",
        help="Path to the log directory",
    )

    parser.add_argument(
        "--script-only",
        action="store_true",
        help="Run script only (no status messages from Heimdallr or Baldr)",
    )

    args = parser.parse_args()

    # logname from the current time

    log_fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logging.basicConfig(
        filename=os.path.join(os.path.expanduser(args.log_location), log_fname),
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add stream handler to also log to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    mcs = MCSClient(
        dcs_endpoints={
            "BLD1": "tcp://192.168.100.2:6662",
            "BLD2": "tcp://192.168.100.2:6663",
            "BLD3": "tcp://192.168.100.2:6664",
            "BLD4": "tcp://192.168.100.2:6665",
            "HDLR": "tcp://192.168.100.2:6660",
        },
        script_endpoint="tcp://192.168.100.2:7019",
        publish_endpoint="tcp://192.168.100.1:7050",
        sleep_time=1.0,
        script_only=args.script_only,
    )

    mcs.run()

if __name__ == "__main__":
    main()

