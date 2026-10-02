# DDSCP fringe servo

Select the new servo with `servo ddscp`. Mode value 5 leaves the existing mode
values unchanged. The controller uses filtered telescope phase delay in K1
wavelengths, predicts three differential piston modes, and sends clipped DM
commands. Its QRD RLS model uses 40 history frames and four future frames.

Exploration lasts for 500 valid DDSCP frames after each mode entry. A missed
frame, lost fringe lock, disconnected phase measurements, an inactive beam, or
a test pattern selects the existing Lacour command path and resets the learned
model. Exploration does not restart after a temporary loss of lock. The model
resumes from the current DM command when four-beam tracking returns.

Run numerical parity tests from `toy-interferometer`:

```sh
uv run --no-sync python -m unittest discover -s ../dcs/heimdallr/tests -p test_predictive_parity.py -q
```

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
