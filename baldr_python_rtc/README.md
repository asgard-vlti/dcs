# Baldr RTC (Python)

Python implementation of the **Baldr real-time controller (RTC)**, originally implemented in C++, we strip back to a python version for ease of testing. We preserve the  existing *commander* ZMQ control interface while modernising the internal architecture for maintainability, testing, and multi-beam operation.

Each Baldr instance runs as a **single process per telescope beam**, exposing a
commander-compatible ZMQ interface and internally managing:
- an RTC loop thread
- a telemetry buffering and writing thread
- a command/control server

This repository currently provides a **functional RTC skeleton** with correct
process structure, commander interface, and telemetry plumbing. Control and
hardware I/O are stubbed and will be incrementally filled in.

---

## Requirements

- Python ≥ 3.10 (tested on 3.11–3.13)
- `pyzmq`
- `numpy`
- `toml` (only required for Python < 3.11)

All dependencies are declared in `pyproject.toml`.

---

## Installation (recommended)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/baldr-rtc-python.git
cd baldr-rtc-python
```
### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```
### 3. Install in editable mode
```bash
python -m pip install -e .
```

### Running the Baldr RTC server

Each Baldr instance corresponds to one beam and one ZMQ endpoint.

By default, the commander socket is derived from the beam number:
	•	beam 1 → tcp://127.0.0.1:3001
	•	beam 2 → tcp://127.0.0.1:3002
	•	…

Start the server (example: beam 1)
```bash
python scripts/baldr_server.py --beam 1 --config ./example_config.toml
```
The server will:
	•	load the configuration
	•	start the RTC loop thread
	•	start the telemetry worker
	•	start the commander ZMQ REP server

From another terminal (with the same virtual environment active) we can test it:
```bash
python scripts/commander_client.py --socket tcp://127.0.0.1:3001 "status"
```
Example commands:
```bash
python scripts/commander_client.py --socket tcp://127.0.0.1:3001 "close_all"
python scripts/commander_client.py --socket tcp://127.0.0.1:3001 "open_all"
python scripts/commander_client.py --socket tcp://127.0.0.1:3001 "pauseRTC"
python scripts/commander_client.py --socket tcp://127.0.0.1:3001 "resumeRTC"
python scripts/commander_client.py --socket tcp://127.0.0.1:3001 "stop_baldr"
```