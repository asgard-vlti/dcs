# DDSPC fringe servo

Select the new servo with `servo "ddspc"`. Mode value 5 leaves the existing mode
values unchanged. The controller uses filtered telescope phase delay in K1
wavelengths, predicts three differential piston modes, and sends clipped DM
commands. Its QRD RLS model uses 40 history frames and four future frames.

The packaged configuration explores for 5000 valid DDSPC frames after each mode entry. A missed
frame, lost fringe lock, disconnected phase measurements, an inactive beam, or
a test pattern selects the existing Lacour command path and resets a model that
is still learning. Exploration does not restart after a temporary loss of lock. The model
resumes from the current DM command when four-beam tracking returns.

The server reads DDSPC defaults from `[ddspc]` in its TOML configuration.
`continue_learning = true` keeps updating the predictive model after exploration.
Set it to `false` to freeze the model and regularization after `n_exploration`
valid frames. A frozen controller still applies its predictive matrix and live
phase-error feedback. If `n_exploration = 0`, it freezes before training.

Use `ddspc "get"` to inspect the configured and active profiles and the active
freeze state. A setter stages one value for the next entry into DDSPC mode; it
does not change an active run.
For example, `ddspc "set-reg-start" 1e5` changes the starting regularization.
The other setters are `set-reg-cutoff`, `set-reg-divisor`, `set-reg-interval`,
`set-n-exploration`, `set-exploration-sigma`, `set-gamma`, and
`set-continue-learning`. For example,
`ddspc "set-continue-learning" false` stages a freeze after exploration.
Commander requires
the quoted action; a comma between the action and value is also accepted.

`ddspc "freeze"` queues an immediate one-way freeze of the active run at the
next control-loop boundary. It also stops exploration dither. It does not change
the configured profile, and returns an error if DDSPC mode is inactive. Repeating
it after a freeze has no effect. `ddspc "get"` reports `freeze_pending` until the
tracking thread applies the request. A frozen model survives a temporary loss
of lock; command and error histories restart from the current DM command.

Regularization is divided every `reg_interval` valid DDSPC frames, starting at
frame index zero, and stops at the hard floor `reg_cutoff`. `gamma` is the QRD RLS
forgetting factor. Exploration sigma is in differential piston waves.

## Model snapshots

Every direct transition from DDSPC to servo off queues one model snapshot. This
includes `servo "off"`, `offload "gd"`, and `offload "mod"`. The command returns
before the background writer finishes; the server logs the saved path or a
write error. Other mode changes do not save a snapshot.

Operational files go to `/data/YYYYMMDD/ddspc_THH:MM:SS.mmm_<pid>_<seq>.json`
using UTC, following the CRED1 date directory and time stamp format. A
simulation build writes under
`~/Documents/0projects/asgard/sim-data/YYYYMMDD/` instead. Files are published
atomically and an existing filename is never overwritten.

Each version 1 JSON file contains the RLS upper triangular factor `R` as a full
249×249 row-major array, its 249×12 weights, the last 12×12 regularized SVD
inverse, and the applied 3×237 predictive matrix. It also records the DDSPC
parameters, actual RLS update count, freeze reason, frame and UTC time, UTC
transition and model times, and whether the
model came from the active or a retained segment. After lost lock or a bad
frame, the last trained segment is retained for the eventual off transition.
An untrained run is still saved, with `inverse` and `predictive` set to JSON
`null`. Non-finite numbers are encoded as `"NaN"`, `"Infinity"`, or
`"-Infinity"` and flagged by `nonfinite_values`.

Run the snapshot checks from the `dcs` directory:

```sh
make -C heimdallr predictive_snapshot
```

Run numerical parity tests from the `dcs` directory:

```sh
python -m unittest discover -s heimdallr/tests -p test_predictive_parity.py -q
```

Run the configuration and controller checks with
`make -C heimdallr predictive_parameters` from the `dcs` directory.

If the toy checkout is elsewhere, set `HEIMDALLR_TOY_DIR` to its directory
before running the test.

Run the deterministic 2 kHz closed-loop diagnostic from the workspace root:

```sh
make -C dcs/heimdallr predictive_simulation
```

The 120,000-update benchmark is available for the instrument-host timing check:

```sh
make -C dcs/heimdallr predictive_benchmark
```

Before operational use, verify the complete fringe loop on the instrument host
at 2 kHz and check its 500 microsecond frame deadline. This verification is
pending while the host is unavailable.
