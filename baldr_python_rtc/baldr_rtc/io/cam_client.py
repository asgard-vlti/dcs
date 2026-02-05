from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict
import zmq

# if you create a simulated camera server it should respond reasonably to the following commands
cred1_command_dict = {
    "all raw": "Display, colon-separated, camera parameters",
    "powers": "Get all camera powers",
    "powers raw": "raw printing",
    "powers getter": "Get getter power",
    "powers getter raw": "raw printing",
    "powers pulsetube": "Get pulsetube power",
    "powers pulsetube raw": "raw printing",
    "temperatures": "Get all camera temperatures",
    "temperatures raw": "raw printing",
    "temperatures motherboard": "Get mother board temperature",
    "temperatures motherboard raw": "raw printing",
    "temperatures frontend": "Get front end temperature",
    "temperatures frontend raw": "raw printing",
    "temperatures powerboard": "Get power board temperature",
    "temperatures powerboard raw": "raw printing",
    "temperatures water": "Get water temperature",
    "temperatures water raw": "raw printing",
    "temperatures ptmcu": "Get pulsetube MCU temperature",
    "temperatures ptmcu raw": "raw printing",
    "temperatures cryostat diode": "Get cryostat temperature from diode",
    "temperatures cryostat diode raw": "raw printing",
    "temperatures cryostat ptcontroller": "Get cryostat temperature from pulsetube controller",
    "temperatures cryostat ptcontroller raw": "raw printing",
    "temperatures cryostat setpoint": "Get cryostat temperature setpoint",
    "temperatures cryostat setpoint raw": "raw printing",
    "fps": "Get frame per second",
    "fps raw": "raw printing",
    "maxfps": "Get the max frame per second regarding current camera configuration",
    "maxfps raw": "raw printing",
    "peltiermaxcurrent": "Get peltiermaxcurrent",
    "peltiermaxcurrent raw": "raw printing",
    "ptready": "Get pulsetube ready information",
    "ptready raw": "raw printing",
    "pressure": "Get cryostat pressure",
    "pressure raw": "raw printing",
    "gain": "Get gain",
    "gain raw": "raw printing",
    "bias": "Get bias correction status",
    "bias raw": "raw printing",
    "flat": "Get flat correction status",
    "flat raw": "raw printing",
    "imagetags": "Get tags in image status",
    "imagetags raw": "raw printing",
    "led": "Get LED status",
    "led raw": "raw printing",
    "sendfile bias <bias image file size> <file MD5>": "Interpreter waits for bias image binary bytes; timeout restarts interpreter.",
    "sendfile flat <flat image file size> <file MD5>": "Interpreter waits for flat image binary bytes.",
    "getflat <url>": "Retrieve flat image from URL.",
    "getbias <url>": "Retrieve bias image from URL.",
    "gettestpattern <url>": "Retrieve test pattern images tar.gz file from URL for testpattern mode.",
    "testpattern": "Get testpattern mode status.",
    "testpattern raw": "raw printing",
    "events": "Camera events sending status",
    "events raw": "raw printing",
    "extsynchro": "Get external synchro usage status",
    "extsynchro raw": "raw printing",
    "rawimages": "Get raw images (no embedded computation) status",
    "rawimages raw": "raw printing",
    "getter nbregeneration": "Get getter regeneration count",
    "getter nbregeneration raw": "raw printing",
    "getter regremainingtime": "Get time remaining for getter regeneration",
    "getter regremainingtime raw": "raw printing",
    "cooling": "Get cooling status",
    "cooling raw": "raw printing",
    "standby": "Get standby mode status",
    "standby raw": "raw printing",
    "mode": "Get readout mode",
    "mode raw": "raw printing",
    "resetwidth": "Get reset width",
    "resetwidth raw": "raw printing",
    "nbreadworeset": "Get read count without reset",
    "nbreadworeset raw": "raw printing",
    "cropping": "Get cropping status (active/inactive)",
    "cropping raw": "raw printing",
    "cropping columns": "Get cropping columns config",
    "cropping columns raw": "raw printing",
    "cropping rows": "Get cropping rows config",
    "cropping rows raw": "raw printing",
    "aduoffset": "Get ADU offset",
    "aduoffset raw": "raw printing",
    "version": "Get all product versions",
    "version raw": "raw printing",
    "version firmware": "Get firmware version",
    "version firmware raw": "raw printing",
    "version firmware detailed": "Get detailed firmware version",
    "version firmware detailed raw": "raw printing",
    "version firmware build": "Get firmware build date",
    "version firmware build raw": "raw printing",
    "version fpga": "Get FPGA version",
    "version fpga raw": "raw printing",
    "version hardware": "Get hardware version",
    "version hardware raw": "raw printing",
    "status": (
        "Get camera status. Possible statuses:\n"
        "- starting: Just after power on\n"
        "- configuring: Reading configuration\n"
        "- poorvacuum: Vacuum between 10-3 and 10-4 during startup\n"
        "- faultyvacuum: Vacuum above 10-3\n"
        "- vacuumrege: Getter regeneration\n"
        "- ready: Ready to be cooled\n"
        "- isbeingcooled: Being cooled\n"
        "- standby: Cooled, sensor off\n"
        "- operational: Cooled, taking valid images\n"
        "- presave: Previous usage error occurred"
    ),
    "status raw": "raw printing",
    "status detailed": "Get last status change reason",
    "status detailed raw": "raw printing",
    "continue": "Resume camera if previously in error/poor vacuum state.",
    "save": "Save current settings; cooling/gain not saved.",
    "save raw": "raw printing",
    "ipaddress": "Display camera IP settings",
    "cameratype": "Display camera information",
    "exec upgradefirmware <url>": "Upgrade firmware from URL",
    "exec buildbias": "Build the bias image",
    "exec buildbias raw": "raw printing",
    "exec buildflat": "Build the flat image",
    "exec buildflat raw": "raw printing",
    "exec redovacuum": "Start vacuum regeneration",
    "set testpattern on": "Enable testpattern mode (loop of 32 images).",
    "set testpattern on raw": "raw printing",
    "set testpattern off": "Disable testpattern mode",
    "set testpattern off raw": "raw printing",
    "set fps <fpsValue>": "Set the frame rate",
    "set fps <fpsValue> raw": "raw printing",
    "set gain <gainValue>": "Set the gain",
    "set gain <gainValue> raw": "raw printing",
    "set bias on": "Enable bias correction",
    "set bias on raw": "raw printing",
    "set bias off": "Disable bias correction",
    "set bias off raw": "raw printing",
    "set flat on": "Enable flat correction",
    "set flat on raw": "raw printing",
    "set flat off": "Disable flat correction",
    "set flat off raw": "raw printing",
    "set imagetags on": "Enable tags in image",
    "set imagetags on raw": "raw printing",
    "set imagetags off": "Disable tags in image",
    "set imagetags off raw": "raw printing",
    "set led on": "Turn on LED; blinks purple if operational.",
    "set led on raw": "raw printing",
    "set led off": "Turn off LED",
    "set led off raw": "raw printing",
    "set events on": "Enable camera event sending (error messages)",
    "set events on raw": "raw printing",
    "set events off": "Disable camera event sending",
    "set events off raw": "raw printing",
    "set extsynchro on": "Enable external synchronization",
    "set extsynchro on raw": "raw printing",
    "set extsynchro off": "Disable external synchronization",
    "set extsynchro off raw": "raw printing",
    "set rawimages on": "Enable embedded computation on images",
    "set rawimages on raw": "raw printing",
    "set rawimages off": "Disable embedded computation",
    "set rawimages off raw": "raw printing",
    "set cooling on": "Enable cooling",
    "set cooling on raw": "raw printing",
    "set cooling off": "Disable cooling",
    "set cooling off raw": "raw printing",
    "set standby on": "Enable standby mode (cools camera, sensor off)",
    "set standby on raw": "raw printing",
    "set standby off": "Disable standby mode",
    "set standby off raw": "raw printing",
    "set mode globalreset": "Set global reset mode (legacy compatibility)",
    "set mode globalresetsingle": "Set global reset mode (single frame)",
    "set mode globalresetcds": "Set global reset correlated double sampling",
    "set mode globalresetbursts": "Set global reset multiple non-destructive readout mode",
    "set mode rollingresetsingle": "Set rolling reset (single frame)",
    "set mode rollingresetcds": "Set rolling reset correlated double sampling (compatibility)",
    "set mode rollingresetnro": "Set rolling reset multiple non-destructive readout",
    "set resetwidth <resetwidthValue>": "Set reset width",
    "set resetwidth <resetwidthValue> raw": "raw printing",
    "set nbreadworeset <nbreadworesetValue>": "Set read count without reset",
    "set nbreadworeset <nbreadworesetValue> raw": "raw printing",
    "set cropping on": "Enable cropping",
    "set cropping on raw": "raw printing",
    "set cropping off": "Disable cropping",
    "set cropping off raw": "raw printing",
    "set cropping columns <columnsValue>": "Set cropping columns selection; format: e.g., '1,3-9'.",
    "set cropping columns <columnsValue> raw": "raw printing",
    "set cropping rows <rowsValue>": "Set cropping rows selection; format: e.g., '1,3,9'.",
    "set cropping rows <rowsValue> raw": "raw printing",
    "set aduoffset <aduoffsetValue>": "Set ADU offset",
    "set aduoffset <aduoffsetValue> raw": "raw printing",
}


def extract_value(s):
    """
    when returning msgs from C-red 1 server they follow a certain format. 
    This function extracts the important bits of the striung
    specifically extracts and returns the substring between the first double quote (")
    and the literal '\\r\\n' sequence from the input string `s`, with surrounding
    whitespace removed.
    
    Parameters:
        s (str): The input string, e.g., '"  1739.356\\r\\nfli-cli>"'
        
    Returns:
        str or None: The extracted substring (with whitespace stripped) if found,
                     otherwise None.
    """

    pattern = r'^"\s*(.*?)\\r\\n'
    match = re.search(pattern, s)
    if match:
        return match.group(1).strip()
    
    return None


class CamClient:
    def __init__(self, host="172.16.8.6", port=6667, timeout_ms=5000, *, context=None):
        self.address = f"tcp://{host}:{port}"
        self._own_ctx = context is None
        self.context = context or zmq.Context.instance()

        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.connect(self.address)

    def send_command(self, command: str, *, pad_to: int | None = None) -> str:
        cmd = command.strip()
        if pad_to is not None and pad_to > 0:
            cmd = cmd.ljust(pad_to)   # pad only; never truncate

        try:
            self.socket.send_string(cmd)
            return self.socket.recv_string()
        except zmq.Again as e:
            raise TimeoutError(f"Camera server timeout for cmd={command!r}") from e
    # def send(self, command: str, cmd_sz: int = 10) -> str:
    #     cmd = command.strip()
    #     if cmd_sz:
    #         # pad or hard-truncate to fixed width (choose one policy)
    #         if len(cmd) < cmd_sz:
    #             cmd = cmd.ljust(cmd_sz)
    #         elif len(cmd) > cmd_sz:
    #             cmd = cmd[:cmd_sz]

    #     try:
    #         self.socket.send_string(cmd)
    #         return self.socket.recv_string()
    #     except zmq.Again as e:
    #         raise TimeoutError(f"Camera server timeout for cmd={command!r}") from e


    def print_camera_commands(self):
        """Prints all available commands and their descriptions in a readable format."""
        print('Available Camera Commands with "send_command()" method:')
        print("=" * 30)
        for command, description in self.command_dict.items():
            print(f"{command}: {description}")
        print("=" * 30)



    def get_camera_config(self) -> Dict:
        # designed for CRED 1 camera 
        # non-exhaustive list that is a good summary of the cred 1 configuration
        keys = [
            "fps",
            "gain",
            "testpattern",
            "bias",
            "flat",
            "imagetags",
            "led",
            "events",
            "extsynchro",
            "rawimages",
            "cooling",
            "mode",
            "resetwidth",
            "nbreadworeset",
            "cropping",
            "cropping columns",
            "cropping rows",
            "aduoffset"
        ]
        config_dict = {}
        for k in keys:
            config_dict[k] = extract_value( self.send_command( f"{k} raw" ) ) # reads the state
        return( config_dict )
     

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self._own_ctx and self.context is not None:
            self.context.term()
            self.context = None


