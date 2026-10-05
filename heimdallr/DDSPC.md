# DDSPC fringe servo

Select the new servo with `servo "ddspc"`. Mode value 5 leaves the existing mode
values unchanged. The controller uses filtered telescope phase delay in K1
wavelengths, predicts three differential piston modes, and sends clipped DM
commands. Its QRD RLS model uses 40 history frames and four future frames.

By default, exploration lasts for 500 valid DDSPC frames after each mode entry. A missed
frame, lost fringe lock, disconnected phase measurements, an inactive beam, or
a test pattern selects the existing Lacour command path and resets the learned
model. Exploration does not restart after a temporary loss of lock. The model
resumes from the current DM command when four-beam tracking returns.

The server reads DDSPC defaults from `[ddspc]` in its TOML configuration.
Use `ddspc "get"` to inspect the configured and active profiles. A setter stages
one value for the next entry into DDSPC mode; it does not change an active run.
For example, `ddspc "set-reg-start" 1e5` changes the starting regularization.
The other setters are `set-reg-cutoff`, `set-reg-divisor`, `set-reg-interval`,
`set-n-exploration`, `set-exploration-sigma`, and `set-gamma`. Commander requires
the quoted action; a comma between the action and value is also accepted.

Regularization is divided every `reg_interval` valid DDSPC frames, starting at
frame index zero, and stops at the hard floor `reg_cutoff`. `gamma` is the QRD RLS
forgetting factor. Exploration sigma is in differential piston waves.

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
