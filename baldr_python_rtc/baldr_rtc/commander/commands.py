from __future__ import annotations
import json
import numpy as np
import threading
import queue
from baldr_python_rtc.baldr_rtc.commander.module import Module, ArgumentSpec
from baldr_python_rtc.baldr_rtc.core.config import readBDRConfig
from baldr_python_rtc.baldr_rtc.core.commands import make_cmd
from baldr_python_rtc.baldr_rtc.core.state import RuntimeGlobals, ServoState, MainState

# parse indicies in commander helper
def _parse_indices(spec: str):
    s = str(spec).strip().lower()
    if s == "all":
        return "all"
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)

# python scripts/commander_client.py --socket tcp://127.0.0.1:3001 'poll_telem {"n":2,"fields":["t_s","frame_id","opd_metric","snr_metric","cmd"]}'
def build_commander_module(
    *,
    globals_: RuntimeGlobals,
    command_queue: "queue.Queue[dict]",
    stop_event: threading.Event,
) -> Module:
    m = Module()

    ##################
    # CONFIG
    def read_bdr(args):
        path = str(args[0]) if args else ""
        cfg = readBDRConfig(path)
        command_queue.put(make_cmd("LOAD_CONFIG", rtc_config=cfg, path=path))
        globals_.active_config_filename = path

        configured = 1 if (len(cfg.matrices.I2M_LO) > 0 or len(cfg.matrices.I2M_HO) > 0) else 0
        return {"ok": True, "config_file": path, "configured": configured, "frequency": cfg.fps}

    ##################
    # TELEMETRY
    def telem_on(args):
        command_queue.put(make_cmd("SET_TELEM", enabled=True))
        return {"ok": True, "take_telemetry": True}

    def telem_off(args):
        command_queue.put(make_cmd("SET_TELEM", enabled=False))
        return {"ok": True, "take_telemetry": False}



    ##################
    # TELEMETRY POLLING (GUI)
    def _ring():
        # tolerate missing ring
        return getattr(globals_, "telem_ring", None) or getattr(globals_, "ring", None)

    def _as_list(x):
        # JSON-safe conversion
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        # fall back (e.g. numpy scalar)
        try:
            return np.asarray(x).tolist()
        except Exception:
            return str(x)

    def _select_fields(chunk, fields, decimate):
        # chunk is TelemetryChunk; fields are attribute names
        out = {}
        for f in fields:
            if not hasattr(chunk, f):
                continue
            v = getattr(chunk, f)
            if v is None:
                out[f] = None
                continue
            if decimate > 1 and isinstance(v, np.ndarray) and v.ndim >= 1:
                v = v[::decimate]
            out[f] = _as_list(v)
        return out

    def poll_telem_info(args):
        r = _ring()
        m = getattr(globals_, "model", None) or getattr(globals_, "rtc_model", None) or getattr(globals_, "SOMETHING", None)
        if r is None:
            return {"ok": False, "error": "no telemetry ring"}

        info = {
            "ok": True,
            "capacity": int(getattr(r, "capacity", 0)),
            "count": int(getattr(r, "_count", 0)),
            "overruns": int(getattr(r, "_overruns", 0)),
        }

        # Try to report configured sizes (these exist in your ring)
        for k in ("n_pix", "n_sig", "n_lo", "n_ho", "n_act"):
            if hasattr(r, k):
                info[k] = int(getattr(r, k))

        # Signal space from model if available
        if m is not None and hasattr(m, "signal_space"):
            info["signal_space"] = str(m.signal_space)

        # Frame shapes if present
        try:
            info["shape_i_raw"] = list(getattr(r, "i_raw").shape[1:])  # (n_pix,) or (H,W) if you later store 2D
        except Exception:
            pass

        return info

    def poll_telem(args):
        """
        args[0] can be:
          - empty: returns last sample of basic fields
          - a JSON string: {"n":50,"fields":[...],"decimate":2}
        """

        r = _ring()
        if r is None:
            return {"ok": False, "error": "no telemetry ring"}

        # defaults: safe small payload
        req = {
            "n": 1,
            "decimate": 1,
            "fields": ["t_s", "opd_metric", "snr_metric", "lo_state", "ho_state", "paused", "frame_id", "overruns"],
        }

        if args:
            a0 = args[0]

            # Case 1: framework already parsed JSON -> dict
            if isinstance(a0, dict):
                req.update(a0)

            # Case 2: user passed fields directly as a python list etc (rare but harmless)
            elif isinstance(a0, list):
                req["fields"] = a0

            # Case 3: string -> try JSON, else try shorthand integer
            else:
                s = str(a0)
                try:
                    user_req = json.loads(s)
                    if isinstance(user_req, dict):
                        req.update(user_req)
                    elif isinstance(user_req, list):
                        req["fields"] = user_req
                except Exception:
                    try:
                        req["n"] = int(s)
                    except Exception:
                        pass



        # # defaults: safe small payload
        # req = {"n": 1, "decimate": 1, "fields": ["t_s", "opd_metric", "snr_metric", "lo_state", "ho_state", "paused", "frame_id", "overruns"]}
        # if args:
        #     s = str(args[0])
        #     try:
        #         user_req = json.loads(s)
        #         if isinstance(user_req, dict):
        #             req.update(user_req)
        #     except Exception:
        #         # allow shorthand: "50" means n=50
        #         try:
        #             req["n"] = int(s)
        #         except Exception:
        #             pass

        n = int(req.get("n", 1))
        decimate = int(req.get("decimate", 1))
        fields = req.get("fields", req["fields"])
        if not isinstance(fields, list):
            fields = req["fields"]

        # clamp n (don’t let GUI accidentally request huge payloads)
        n = max(1, min(n, 2000))
        decimate = max(1, min(decimate, 128))

        chunk = r.snapshot_latest(n=n)

        # Always include overruns in response (even if not requested)
        payload = _select_fields(chunk, fields, decimate)
        if "overruns" not in payload:
            payload["overruns"] = int(getattr(chunk, "overruns", 0))

        # Convenience: include n returned after decimation
        # (t_s exists if requested; otherwise estimate from first array in payload)
        n_ret = None
        for v in payload.values():
            if isinstance(v, list):
                n_ret = len(v)
                break

        return {"ok": True, "n": int(n_ret or 0), "fields": payload}
    ###################
    # RTC STATE
    def pause_rtc(args):
        command_queue.put(make_cmd("PAUSE"))
        return {"ok": True}

    def resume_rtc(args):
        command_queue.put(make_cmd("RESUME"))
        return {"ok": True}

    def stop_baldr(args):
        command_queue.put(make_cmd("STOP"))
        return {"ok": True, "servo_mode": int(MainState.SERVO_STOP)}


    # DIONT USE #################
    # GAIN
    # def set_LO_gain(args):
    #     gains = [float(aa) for aa in args]
    #     command_queue.put(make_cmd("SET_LO_GAIN", gain=gains))
    #     return {"ok": True, "LO_gain": gains}

    # def set_HO_gain(args):
    #     gains = [float(aa) for aa in args]
    #     command_queue.put(make_cmd("SET_HO_GAIN", gain=gains))
    #     return {"ok": True, "HO_gain": gains}

    
    ###################
    # CONTROLLER STATE
    def close_all(args):
        command_queue.put(make_cmd("SET_LOHO", lo=int(ServoState.SERVO_CLOSE), ho=int(ServoState.SERVO_CLOSE)))
        return {"ok": True}

    def open_all(args):
        command_queue.put(make_cmd("SET_LOHO", lo=int(ServoState.SERVO_OPEN), ho=int(ServoState.SERVO_OPEN)))
        return {"ok": True}

    def close_lo(args):
        command_queue.put(make_cmd("SET_LO", value=int(ServoState.SERVO_CLOSE)))
        return {"ok": True}

    def open_lo(args):
        command_queue.put(make_cmd("SET_LO", value=int(ServoState.SERVO_OPEN)))
        return {"ok": True}

    def close_ho(args):
        command_queue.put(make_cmd("SET_HO", value=int(ServoState.SERVO_CLOSE)))
        return {"ok": True}

    def open_ho(args):
        command_queue.put(make_cmd("SET_HO", value=int(ServoState.SERVO_OPEN)))
        return {"ok": True}

    # #################
    # GAIN
    # setting gains 
    def set_lo_gain_cmd(args):
        # set_lo_gain <param> <indices|all> <value>
        if len(args) < 3:
            return {"ok": False, "error": "usage: set_lo_gain <param> <indices|all> <value>"}

        param = str(args[0]).strip().lower()
        idx_spec = args[1]
        value = float(args[2])

        if param not in ("kp", "ki", "kd", "rho"):
            return {"ok": False, "error": f"invalid param '{param}', expected kp|ki|kd|rho"}

        idx = _parse_indices(idx_spec)

        command_queue.put(make_cmd("SET_LO_GAIN", param=param, idx=idx, value=value))
        return {"ok": True, "param": param, "idx": idx, "value": value}

    def set_ho_gain_cmd(args):
        # set_ho_gain <param> <indices|all> <value>
        if len(args) < 3:
            return {"ok": False, "error": "usage: set_ho_gain <param> <indices|all> <value>"}

        param = str(args[0]).strip().lower()
        idx_spec = args[1]
        value = float(args[2])

        if param not in ("kp", "ki", "kd", "rho"):
            return {"ok": False, "error": f"invalid param '{param}', expected kp|ki|kd|rho"}

        idx = _parse_indices(idx_spec)

        command_queue.put(make_cmd("SET_HO_GAIN", param=param, idx=idx, value=value))
        return {"ok": True, "param": param, "idx": idx, "value": value}

    def zero_gains_cmd(args):
        command_queue.put(make_cmd("ZERO_GAINS"))
        return {"ok": True}
    ################## 
    # UPDATE REFERENCE INTENSITIES 
    def update_N0_runtime(args): 
        command_queue.put(make_cmd("UPDATE_N0_RUNTIME"))
        return {"ok": True}
    
    def update_I0_runtime(args): 
        command_queue.put(make_cmd("UPDATE_I0_RUNTIME"))
        return {"ok": True}
    ##################
    # STATUS
    # !!IMPORTANT!! - this format is to match the expected format for wag (see legacy get_status in https://github.com/mikeireland/dcs/blob/main/baldr/baldr.cpp)
    def status(args):
        cfg = globals_.rtc_config
        lo_ready = len(cfg.matrices.I2M_LO) > 0
        ho_ready = len(cfg.matrices.I2M_HO) > 0
        configured = 1 if (lo_ready or ho_ready) else 0

        return {
            "TT_state": int(globals_.servo_mode_LO),
            "HO_state": int(globals_.servo_mode_HO),
            "mode": globals_.mode or "unknown", 
            "phasemask": globals_.phasemask or "unknown",
            "frequency": float(cfg.fps),
            "configured": int(configured),
            "ctrl_type": cfg.state.controller_type,
            "config_file": globals_.active_config_filename or "unknown",
            "inj_enabled": 1 if cfg.inj_signal.enabled else 0,
            "auto_loop": 1 if cfg.state.auto_close else 0,
            "close_on_strehl": float(cfg.limits.close_on_strehl_limit),
            "open_on_strehl": float(cfg.limits.open_on_strehl_limit),
            "close_on_snr": 2.0,
            "open_on_snr": float(cfg.limits.open_on_flux_limit),
            "TT_offsets": 0,
        }


    ##################
    # CONFIG
    m.def_command("readBDRConfig", read_bdr, description="Load/parse config.", arguments=[ArgumentSpec("config_file", "string")], return_type="object")
    ##################
    # TELEMETRY
    m.def_command("telem_on", telem_on, description="Enable telemetry writing.", return_type="object")
    m.def_command("telem_off", telem_off, description="Disable telemetry writing.", return_type="object")
    ###################
    # RTC STATE
    m.def_command("pauseRTC", pause_rtc, description="Pause RTC loop.", return_type="object")
    m.def_command("resumeRTC", resume_rtc, description="Resume RTC loop.", return_type="object")
    m.def_command("stop_baldr", stop_baldr, description="Stop Baldr RTC loop.", return_type="object")
    ###################
    # CONTROLLER STATE
    m.def_command("close_all", close_all, description="Close LO+HO loops.", return_type="object")
    m.def_command("open_all", open_all, description="Open LO+HO loops.", return_type="object")
    m.def_command("close_baldr_LO", close_lo, description="Close LO loop.", return_type="object")
    m.def_command("open_baldr_LO", open_lo, description="Open LO loop.", return_type="object")
    m.def_command("close_baldr_HO", close_ho, description="Close HO loop.", return_type="object")
    m.def_command("open_baldr_HO", open_ho, description="Open HO loop.", return_type="object")
    
    
    m.def_command(
        "zero_gains",
        zero_gains_cmd,
        description="Zero all controller gains (kp/ki/kd) for LO and HO.",
        return_type="object",
    )
    
    ##################
    # STATUS
    m.def_command("status", status, description="Get Baldr status snapshot.", return_type="object")
    
    ##################
    # UPDATE REFERENCE INTENSITIES
    m.def_command("update_N0_runtime", update_N0_runtime, description="Update clear pupil normalization in runtime", return_type="object") 
    m.def_command("update_I0_runtime", update_I0_runtime, description="Update ZWFS pupil intensity setpoint", return_type="object") 

    ##################
    # GAIN

    # m.def_command("set_LO_gain", set_LO_gain, description="Update LO gains", return_type="object") 
    # m.def_command("set_HO_gain", set_HO_gain, description="Update HO gains", return_type="object") 


    m.def_command(
        "set_lo_gain",
        set_lo_gain_cmd,
        description="Set LO gain: set_lo_gain <kp|ki|kd|rho> <all|idxspec> <value>",
        arguments=[
            ArgumentSpec("param", "string"),
            ArgumentSpec("indices", "string"),
            ArgumentSpec("value", "string"),
        ],
        return_type="object",
    )

    m.def_command(
        "set_ho_gain",
        set_ho_gain_cmd,
        description="Set HO gain: set_ho_gain <kp|ki|kd|rho> <all|idxspec> <value>",
        arguments=[
            ArgumentSpec("param", "string"),
            ArgumentSpec("indices", "string"),
            ArgumentSpec("value", "string"),
        ],
        return_type="object",
    )

    ##################
    # TELEMETRY POLL
    m.def_command("poll_telem", poll_telem, description="Poll latest telemetry samples (JSON request).", arguments=[ArgumentSpec("request_json_or_n", "string")], return_type="object")
    m.def_command("poll_telem_info", poll_telem_info, description="Telemetry ring info/sizes.", return_type="object")
    
    return m


### SOME EXAMPLES 
# POLLING TELEMETRY (USEFUL FOR A GUI)
# python scripts/commander_client.py --socket tcp://127.0.0.1:3001 'poll_telem {"n":2,"fields":["t_s","frame_id","opd_metric","snr_metric","cmd"]}'