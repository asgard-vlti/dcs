# Change the Baldr mode from STANDARD to FAINT or back.

import zmq
import dcs.ZMQutils
import sys
import json

class BMode:
    def __init__(self, mode):

        self.mds = self._open_mds_connection()

        self.mode = mode


        self.mcs_client = dcs.ZMQutils.ZmqReq("tcp://192.168.100.2:7019")

    def _open_mds_connection(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        server_address = "tcp://192.168.100.2:5555"
        socket.connect(server_address)
        return socket

    def send_and_recv_ack(self, msg):
        # recieve ack
        print(f"sending {msg}")
        resp = self.mcs_client.send_payload(msg, decode_ascii=False)
        if resp is None or resp.get("ok") == False:
            print(resp)
            print("Failed to send complete to MCS")
        else:
            print("msg acked")

    def _send_and_get_response(self, message):
        # print("sending", message)
        self.mds.send_string(message)
        response = self.mds.recv_string()
        # print("response", response)
        return response.strip()

    def _move_all_BLF_beams(self, target_pos: str):
        results = []
        for beam_num in [1,2,3,4]:
            target = f"BLF{beam_num}"
            message = f"asg_setup {target} NAME {target_pos}"
            try:
                res = self._send_and_get_response(message)
            except Exception as e:
                res = f"ERROR: {e}"
            results.append((beam_num, message, res))
        return results

    def run(self):
        if self.mode == "FAINT":
            # Implementation for FAINT mode
            file_pth = "/home/asg/.config/asgard-alignment/stable_states/baldr_ONLY_faint.json"
            self._move_all_BLF_beams("FAINT")
        elif self.mode == "STANDARD":
            # Implementation for STANDARD mode
            file_pth = "/home/asg/.config/asgard-alignment/stable_states/baldr_ONLY_bright.json"
            self._move_all_BLF_beams("STANDARD")
        else:
            raise ValueError("Invalid mode. Please specify either 'FAINT' or 'STANDARD'.")
        print( f"loading hard coded stable state from {file_pth}")
        with open(file_pth) as f:
            states = json.load(f)

            for state in states:
                if state["is_connected"]:
                    if "BLF" in state["name"]:
                        # BLF can't tell where it is, so do nothing
                        print("passing on BLF")
                        pass
                    else:
                        message = (
                                    f"moveabs {state['name']} {state['position']}"
                                )
                        self._send_and_get_response(message)

def main():
    # Use argv to determine the mode (FAINT or STANDARD) and other parameters
    if len(sys.argv) != 2:
        print("Usage: python b_mode.py [FAINT|STANDARD]")
        sys.exit(1)
    mode = sys.argv[1].upper()
    if mode not in ["FAINT", "STANDARD"]:
        print("Invalid mode. Please specify either 'FAINT' or 'STANDARD'.")
        sys.exit(1)
    shutter_seq = BMode(mode)
    shutter_seq.run()

if __name__ == "__main__":    
    main()