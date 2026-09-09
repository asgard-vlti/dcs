# baldr RTC

Jesse's implementation of the baldr RTC.

## Quick Reference

For the simulator, the bench, and on-sky, the following supervisor
commands should work well enough:

To init on beam 1 (reset all matrices and control variables):
```bash
python supervisor.py 1 --init
```

To reset (reset control variables online):
```bash
python supervisor.py 1 --reset
```

To perform a poke test, to measure interaction matrix, and to compute and set the control matrices:
```bash
python supervisor.py 1 --recompute
```

To perform a poke test with a specific poke (default=0.1) and/or number of modes (default=max=100) to control:
```bash
python supervisor.py 1 --recompute --poke 0.01 --nmodes 50
```

To set leaky integrator gain and leak:
```bash
python supervisor.py 1 --gain 0.3 --leak 0.99
```

To open the loop:
```bash
python supervisor.py 1 --gain 0.0 --leak 0.0
```


## Installation

In the `dcs` root directory, follow the `cmake` instructions in the `README.md`. The executable
will be built to `./build/baldr_jcr/baldr`.

## Running

The program is split into an **RTC**, and a **supervisor**.

The RTC expects many arrays to exist on disk, but those files are not committed to
the git repository. To generate them with default values, run (from this directory):

```bash
python supervisor.py 1 --init  # initialise beam 1 arrays
```

After initialising these arrays, you should be able to launch the RTC using the `baldr`
command, for example:

```bash
./build/baldr_jcr/baldr ./baldr_jcr/def1.toml --socket=tcp://localhost:6662
```

With the RTC running, you can interact with it via the supervisor. To see a list
of available supervisor commands, run:

```bash
python supervisor.py --help
```

For example, to recompute the interaction and control matrices for a leaky integrator controller
with a gain of 0.3 and leak of 0.999, run:

```bash
python supervisor.py 1 --leaky --recompute --gain=0.3 --leak=0.999
```

## Todo:

- [ ] URGENT: refactor to allow one RTC instance per beam, currently there will be naming
      collisions.
- [ ] replace "doubles" everywhere in RTC with singles, way overkill and should bump
      performance a little
- [ ] align datatypes in Diagram with implementation, since Diagram should be the truth "reference"

## RTC Logic

![rtc diagram](./baldr_control_logic.svg)

## Internal Compliance

### Data Object Implementation

| Data Object         | Diagram | c-header | servo loop | baldr | config | Commander | supervisor |
| ------------------- | :-----: | :------: | :--------: | :---: | :----: | :-------: | :--------: |
| `meas_offset`\*     |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `meas_offset_lut`\* |   :x:   |   :x:    |    :x:     |  :x:  |  :x:   |    :x:    |    :x:     |
| `flux_mask`         |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:     |
| `strehl_mask`       |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:     |
| `meas_to_mode`      |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `filter_coeff_in`   |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `filter_coeff_out`  |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `mode_offset`       |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `mode_max`          |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `mode_min`          |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `mode_to_com`       |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `com_max`           |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `com_min`           |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `com_dist_buffer`   |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |

\* note that meas_offset_lut should supercede meas_offset.

### HRTC Pipeline

| Task                | Implemented | Tested |
| ------------------- | :---------: | :----: |
| `read_shm`          |    :ok:     |  :x:   |
| `calibrate_frame`   |    :ok:     |  :x:   |
| `compute_pol_meas`  |    :ok:     |  :x:   |
| `reconstruct_modes` |    :ok:     |  :x:   |
| `filter_modes`      |    :ok:     |  :x:   |
| `project_com`       |    :ok:     |  :x:   |
| `clip_com`          |    :ok:     |  :x:   |
| `inject_disturb`    |    :ok:     |  :x:   |
| `write_shm`         |    :ok:     |  :x:   |
