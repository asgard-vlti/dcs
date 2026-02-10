# #!/usr/bin/env python3

# client_run_bautoalign.py
from __future__ import annotations
import json
import datetime as dt
import zmq

def send_command(server: str, name: str, parameters: dict | list | None = None) -> dict:
    """
    Sends a single JSON command to the BackEndServer and returns the reply dict.
    - server: "tcp://<host>:<port>", e.g., "tcp://127.0.0.1:7002"
    - name:   command["name"], e.g., "s_b-autoalign", "ping", "setup"
    - parameters: either a dict {"key": value, ...} OR a list of {"name","value"} dicts
    """
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    # timeouts so we don't hang forever
    s.setsockopt(zmq.RCVTIMEO, 5000)   # 5s receive timeout
    s.setsockopt(zmq.SNDTIMEO, 5000)   # 5s send timeout
    s.connect(server)

    msg = {
        "command": {
            "name": name,
            "time": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    }
    if parameters:
        if isinstance(parameters, dict):
            msg["command"]["parameters"] = [{"name": k, "value": v} for k, v in parameters.items()]
        else:
            # assume already in RTS list form
            msg["command"]["parameters"] = parameters

    s.send_string(json.dumps(msg))
    reply = s.recv_json()
    return reply

def run_bautoalign(server: str = "tcp://127.0.0.1:7002") -> None:
    """Call the server to run the Baldr auto-align script."""
    rep = send_command(server, name="s_b-autoalign")
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    # hit the default on localhost:7002
    run_bautoalign()



####################### THis is all outdated testes below .. ignore 
# # test_back_end_client.py
# import argparse
# import datetime as dt
# import json
# import sys
# import time
# import threading
# import zmq

# DEF_HOST = "localhost"
# DEF_PORT = 7004
# DEF_TIMEOUT_MS = 5000  # client-side REQ timeout

# # ---------------- helpers ----------------
# def now_iso():
#     return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

# def type_cast(v: str):
#     vl = v.lower()
#     if vl in ("true", "false"):
#         return vl == "true"
#     try:
#         return int(v)
#     except ValueError:
#         pass
#     try:
#         return float(v)
#     except ValueError:
#         return v

# def parse_kv_params(kvs):
#     """['k=v', ...] -> [{'name': k, 'value': typed(v)}, ...]"""
#     out = []
#     for kv in (kvs or []):
#         if "=" not in kv:
#             print(f"ignoring param '{kv}' (need key=value)", file=sys.stderr)
#             continue
#         k, v = kv.split("=", 1)
#         out.append({"name": k, "value": type_cast(v)})
#     return out

# def build_message(name, parameters=None):
#     return {
#         "command": {
#             "name": name,
#             "time": now_iso(),
#             "parameters": parameters or []
#         }
#     }

# def send_once(host, port, message, timeout_ms=DEF_TIMEOUT_MS):
#     ctx = zmq.Context.instance()
#     sock = ctx.socket(zmq.REQ)
#     try:
#         sock.setsockopt(zmq.LINGER, 0)
#         sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
#         sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
#         sock.connect(f"tcp://{host}:{port}")
#         sock.send_string(json.dumps(message))
#         return sock.recv_json()
#     finally:
#         sock.close()

# def pretty(obj):
#     print(json.dumps(obj, indent=2, sort_keys=True))

# # ---------------- command funcs ----------------
# def cmd_ping(args):
#     msg = build_message("ping")
#     pretty(send_once(args.host, args.port, msg, args.timeout))

# def cmd_setup(args):
#     params = parse_kv_params(args.param)
#     if args.dit is not None:
#         params.append({"name": "DET.DIT", "value": float(args.dit)})
#     if args.ndit is not None:
#         params.append({"name": "DET.NDIT", "value": int(args.ndit)})
#     if args.nworeset is not None:
#         params.append({"name": "DET.NWORESET", "value": int(args.nworeset)})
#     if args.gain is not None:
#         params.append({"name": "DET.GAIN", "value": int(args.gain)})
#     msg = build_message("setup", params)
#     pretty(send_once(args.host, args.port, msg, args.timeout))

# def cmd_start(args):
#     params = parse_kv_params(args.param)
#     msg = build_message("start", params)
#     pretty(send_once(args.host, args.port, msg, args.timeout))

# def cmd_abort_readout(args):
#     params = parse_kv_params(args.param)
#     msg = build_message("abort", params)
#     pretty(send_once(args.host, args.port, msg, args.timeout))

# def cmd_expstatus(args):
#     msg = build_message("expstatus")
#     pretty(send_once(args.host, args.port, msg, args.timeout))

# def cmd_report_jobs(args):
#     job_id = getattr(args, "job_id", None)
#     timeout = getattr(args, "timeout", DEF_TIMEOUT_MS)
#     params = [{"name":"job_id","value": job_id}] if job_id else []
#     msg = build_message("report_jobs", params)
#     pretty(send_once(args.host, args.port, msg, timeout))
#     # params = [{"name": "job_id", "value": args.job_id}] if args.job_id else []
#     # msg = build_message("report_jobs", params)
#     # pretty(send_once(args.host, args.port, msg, args.timeout))

# def cmd_abort_job(args):
#     if not args.job_id:
#         print("abort_job requires --job-id", file=sys.stderr)
#         sys.exit(2)
#     params = [{"name": "job_id", "value": args.job_id}]
#     msg = build_message("abort", params)
#     pretty(send_once(args.host, args.port, msg, args.timeout))

# def cmd_rts(args):
#     # Determine wire name; default to prefix 'bld_' unless already provided or --raw
#     name = args.name
#     if not args.raw and not name.startswith("bld_"):
#         name = f"bld_{name}"

#     params = parse_kv_params(args.param)
#     if args.beam:
#         params.append({"name": "beamid", "value": args.beam})
#     msg = build_message(name, params)
#     pretty(send_once(args.host, args.port, msg, args.timeout))

# # ---- burst helpers ----
# def _send_rts_thread(host, port, timeout, name, idx, beam, raw):
#     # each thread uses its own REQ socket
#     wire_name = name
#     if not raw and not wire_name.startswith("bld_"):
#         wire_name = f"bld_{wire_name}"
#     params = [
#         {"name": "beamid", "value": beam},   # required by your Baldr handler
#         {"name": "idx", "value": idx},       # optional, helpful while testing
#     ]
#     msg = build_message(wire_name, params)
#     try:
#         reply = send_once(host, port, msg, timeout)
#         print(f"[{idx}] {reply['reply']['content']}")
#     except Exception as e:
#         print(f"[{idx}] ERROR: {e}", file=sys.stderr)

# def cmd_burst(args):
#     threads = []
#     for i in range(args.n):
#         t = threading.Thread(
#             target=_send_rts_thread,
#             args=(args.host, args.port, args.timeout, args.name, i, args.beam, args.raw),
#             daemon=True,
#         )
#         t.start()
#         threads.append(t)
#         if args.stagger > 0:
#             time.sleep(args.stagger)
#     for t in threads:
#         t.join()
#     # show snapshot
#     cmd_report_jobs(args)

# # Optional: Baldr shortcut subcommand (always prefixes bld_)
# def cmd_baldr(args):
#     ns = argparse.Namespace(
#         host=args.host, port=args.port, timeout=args.timeout,
#         name=args.name, raw=False, beam=args.beam, param=None
#     )
#     return cmd_rts(ns)

# # ---------------- main ----------------
# def main():
#     ap = argparse.ArgumentParser(description="Basic client for BackEndServer (ZMQ REQ).")
#     ap.add_argument("--host", default=DEF_HOST)
#     ap.add_argument("--port", type=int, default=DEF_PORT)
#     ap.add_argument("--timeout", type=int, default=DEF_TIMEOUT_MS, help="ms REQ/REP timeout")

#     sub = ap.add_subparsers(dest="cmd", required=False)

#     sp = sub.add_parser("ping", help="send PING")
#     sp.set_defaults(func=cmd_ping)

#     ssetup = sub.add_parser("setup", help="send SETUP (e.g., DET.* params)")
#     ssetup.add_argument("--param", action="append", help="key=value (repeatable)")
#     ssetup.add_argument("--dit", type=float, help="maps to DET.DIT (seconds)")
#     ssetup.add_argument("--ndit", type=int, help="maps to DET.NDIT")
#     ssetup.add_argument("--nworeset", type=int, help="maps to DET.NWORESET")
#     ssetup.add_argument("--gain", type=int, help="maps to DET.GAIN (0–100)")
#     ssetup.set_defaults(func=cmd_setup)

#     sstart = sub.add_parser("start", help="send START")
#     sstart.add_argument("--param", action="append", help="key=value (optional)")
#     sstart.set_defaults(func=cmd_start)

#     sabort_ro = sub.add_parser("abort_readout", help="send ABORT (readout abort)")
#     sabort_ro.add_argument("--param", action="append", help="key=value (optional)")
#     sabort_ro.set_defaults(func=cmd_abort_readout)

#     sexps = sub.add_parser("expstatus", help="send EXPSTATUS")
#     sexps.set_defaults(func=cmd_expstatus)

#     sj = sub.add_parser("report_jobs", help="list queued/running/done jobs")
#     sj.add_argument("--job-id", help="filter a single job id")
#     sj.set_defaults(func=cmd_report_jobs)

#     sa = sub.add_parser("abort_job", help="abort a running RTS job by id")
#     sa.add_argument("--job-id", required=True)
#     sa.set_defaults(func=cmd_abort_job)

#     sr = sub.add_parser("rts", help="send an RTS command (defaults to bld_<name>)")
#     sr.add_argument("name", help="RTS name, e.g. open_baldr_lo")
#     sr.add_argument("--raw", action="store_true", help="do not prefix with 'bld_'")
#     sr.add_argument("--beam", help="beam id: 1..4, 0=all, or comma list '1,3'")
#     sr.add_argument("--param", action="append", help="additional key=value (repeatable)")
#     sr.set_defaults(func=cmd_rts)

#     sb = sub.add_parser("burst", help="fire N RTS commands quickly")
#     sb.add_argument("name", help="RTS name, e.g. open_baldr_lo")
#     sb.add_argument("-n", type=int, default=10, help="number of RTS to send")
#     sb.add_argument("--stagger", type=float, default=0.0, help="seconds between sends")
#     sb.add_argument("--beam", required=True, help="beam id: 1..4, 0=all, or comma list")
#     sb.add_argument("--raw", action="store_true", help="do not prefix with 'bld_'")
#     sb.set_defaults(func=cmd_burst)

#     # Optional baldr convenience: always prefixes bld_ and requires --beam
#     sbd = sub.add_parser("baldr", help="Baldr RTS shortcut (always bld_<name>)")
#     sbd.add_argument("name", choices=[
#         "open_baldr_lo","close_baldr_lo",
#         "open_baldr_ho","close_baldr_ho",
#         "open_all","close_all","save_telemetry"
#     ])
#     sbd.add_argument("--beam", required=True, help="beam id: 1..4, 0=all, or comma list")
#     sbd.set_defaults(func=cmd_baldr)

#     args = ap.parse_args()

#     if not args.cmd:
#         # Default mini-sequence: ping → open_baldr_lo (beam 1) → report_jobs
#         print("-> ping")
#         cmd_ping(args)
#         print("-> bld_open_baldr_lo (beam 1)")
#         args2 = argparse.Namespace(**vars(args), name="open_baldr_lo", raw=False, beam="1", param=None, cmd="rts")
#         cmd_rts(args2)
#         print("-> report_jobs")
#         cmd_report_jobs(args)
#         return

#     args.func(args)

# if __name__ == "__main__":
#     main()



# """
# # point to your non-conflicting port
# python test_back_end_client.py --port 7010 ping

# # SETUP (1 kHz, no save, NWORESET=2, gain=10)
# python test_back_end_client.py --port 7010 setup --dit 0.001 --ndit 0 --nworeset 2 --gain 10

# # START → EXPSTATUS → ABORT readout
# python test_back_end_client.py --port 7010 start
# python test_back_end_client.py --port 7010 expstatus
# python test_back_end_client.py --port 7010 abort_readout

# # Baldr RTS (beam required)
# python test_back_end_client.py --port 7010 rts open_baldr_lo --beam 2
# python test_back_end_client.py --port 7010 rts close_all --beam 0
# python test_back_end_client.py --port 7010 rts save_telemetry --beam 1

# # Burst test (bounded pool exercise)
# python test_back_end_client.py --port 7010 burst open_baldr_lo --beam 1 -n 20 --stagger 0.05


# # Jobs + abort a specific job
# python test_back_end_client.py --port 7010 report_jobs
# python test_back_end_client.py --port 7010 abort_job --job-id <paste-id>
# """

# # # test_back_end_client.py
# # import argparse
# # import datetime as dt
# # import json
# # import sys
# # import time
# # import zmq
# # import threading

# # DEF_HOST = "localhost"
# # DEF_PORT = 7010

# # def now_iso():
# #     return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

# # def build_message(name, parameters=None):
# #     return {
# #         "command": {
# #             "name": name,
# #             "time": now_iso(),
# #             "parameters": parameters or []
# #         }
# #     }

# # def send_once(host, port, message):
# #     ctx = zmq.Context.instance()
# #     sock = ctx.socket(zmq.REQ)
# #     sock.connect(f"tcp://{host}:{port}")
# #     sock.send_string(json.dumps(message))
# #     reply = sock.recv_json()
# #     sock.close()
# #     return reply

# # def pretty(reply):
# #     print(json.dumps(reply, indent=2, sort_keys=True))

# # def cmd_ping(args):
# #     msg = build_message("ping")
# #     pretty(send_once(args.host, args.port, msg))

# # def cmd_rts(args):
# #     # accepts either "name" or "bld_name". both work with your server.
# #     rts_name = args.name if args.raw else f"bld_{args.name}"
# #     params = []
# #     for kv in (args.param or []):
# #         # --param key=value
# #         if "=" not in kv:
# #             print(f"ignoring param '{kv}' (need key=value)", file=sys.stderr)
# #             continue
# #         k, v = kv.split("=", 1)
# #         # best-effort type parsing (int/float), else string
# #         try:
# #             if v.lower() in ("true","false"):
# #                 val = v.lower() == "true"
# #             else:
# #                 val = int(v)
# #         except ValueError:
# #             try:
# #                 val = float(v)
# #             except ValueError:
# #                 val = v
# #         params.append({"name": k, "value": val})
# #     msg = build_message(rts_name, params)
# #     pretty(send_once(args.host, args.port, msg))

# # def cmd_report_jobs(args):
# #     params = [{"name": "job_id", "value": args.job_id}] if args.job_id else []
# #     msg = build_message("report_jobs", params)
# #     pretty(send_once(args.host, args.port, msg))

# # def cmd_abort(args):
# #     if not args.job_id:
# #         print("abort requires --job-id", file=sys.stderr)
# #         sys.exit(2)
# #     params = [{"name": "job_id", "value": args.job_id}]
# #     msg = build_message("abort", params)
# #     pretty(send_once(args.host, args.port, msg))

# # def _send_rts_thread(host, port, name, idx):
# #     # each thread uses its own REQ socket (REQ is not thread-safe)
# #     rts = f"bld_{name}"
# #     msg = build_message(rts, [{"name":"idx","value":idx}])
# #     try:
# #         reply = send_once(host, port, msg)
# #         print(f"[{idx}] {reply['reply']['content']}")
# #     except Exception as e:
# #         print(f"[{idx}] ERROR: {e}", file=sys.stderr)

# # def cmd_burst(args):
# #     """Fire N RTS commands quickly to exercise the server’s bounded pool."""
# #     threads = []
# #     for i in range(args.n):
# #         t = threading.Thread(target=_send_rts_thread, args=(args.host, args.port, args.name, i), daemon=True)
# #         t.start()
# #         threads.append(t)
# #         if args.stagger > 0:
# #             time.sleep(args.stagger)
# #     for t in threads:
# #         t.join()
# #     # show snapshot
# #     cmd_report_jobs(args)

# # def main():
# #     ap = argparse.ArgumentParser(description="Basic client for BackEndServer (ZMQ REQ).")
# #     ap.add_argument("--host", default=DEF_HOST)
# #     ap.add_argument("--port", type=int, default=DEF_PORT)

# #     sub = ap.add_subparsers(dest="cmd", required=False)

# #     sp = sub.add_parser("ping", help="send ping")
# #     sp.set_defaults(func=cmd_ping)

# #     sr = sub.add_parser("rts", help="send an RTS command (defaults to bld_<name>)")
# #     sr.add_argument("name", help="RTS name, e.g. openao")
# #     sr.add_argument("--raw", action="store_true", help="do not prefix with 'bld_'")
# #     sr.add_argument("--param", action="append", help="key=value (repeatable)")
# #     sr.set_defaults(func=cmd_rts)

# #     sj = sub.add_parser("report_jobs", help="list queued/running/done jobs")
# #     sj.add_argument("--job-id", help="filter a single job id")
# #     sj.set_defaults(func=cmd_report_jobs)

# #     sa = sub.add_parser("abort", help="abort a running job by id")
# #     sa.add_argument("--job-id", required=True)
# #     sa.set_defaults(func=cmd_abort)

# #     sb = sub.add_parser("burst", help="fire N RTS commands quickly")
# #     sb.add_argument("name", help="RTS name, e.g. openao")
# #     sb.add_argument("-n", type=int, default=10, help="number of RTS to send")
# #     sb.add_argument("--stagger", type=float, default=0.0, help="seconds between sends")
# #     sb.set_defaults(func=cmd_burst)

# #     args = ap.parse_args()

# #     if not args.cmd:
# #         # default mini-sequence: ping - bld_openao - report_jobs
# #         print("-> ping")
# #         cmd_ping(args)
# #         print("-> bld_close_baldr_lo")
# #         args2 = argparse.Namespace(**vars(args), name="openao", raw=False, param=None)
# #         cmd_rts(args2)
# #         print("-> report_jobs")
# #         cmd_report_jobs(args)
# #         return

# #     args.func(args)

# # if __name__ == "__main__":
# #     main()


# # """
# # python test_back_end_client.py

# # or for particular tests:
# # python test_back_end_client.py ping
# # python test_back_end_client.py rts openao --param beam=1
# # python test_back_end_client.py report_jobs
# # python test_back_end_client.py abort --job-id <id>
# # python test_back_end_client.py burst openao -n 20 --stagger 0.05

# # """