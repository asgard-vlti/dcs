# baldr RTC

Jesse's implementation of the baldr RTC.

## **@MIKE** INSTRUCTIONS FOR SETTING UP RTC (10TH SEPTEMBER 2026):

### RTC

Once the `baldr` executable is built, place it somewhere on your PATH (e.g., `/usr/bin/local/baldr`). See [installation](#installation) for build instructions if you run into trouble.

If you are starting `baldr` manually, you need to set the `BALDR_ROOT` environment
variable, e.g.,

```bash
export BALDR_ROOT="/usr/local/etc"
```

Then you can start `baldr` for each beam, e.g.:

```bash
# start beam 1 rtc
baldr /usr/local/etc/def1.toml --socket=tcp://localhost:6662
```

Note: If you are starting `baldr` using the `./run_scripts/run_baldr`, the BALDR_ROOT
variable will be overridden to the system setting (usually `/usr/local/etc`).

### Supervisor

A single python script acts as the "supervisor", allowing interaction with the
RTC using Commander and ZMQ under the hood. It's recommended to run the
supervisor script from its directory, but that may not be strictly necessary.

```bash
cd ./baldr_jcr
./supervisor.py --help
```

The `BALDR_ROOT` environment variable should be set to match the one used when
launching the RTC. In production mode, this should be:

```bash
export BALDR_ROOT=/usr/local/bin
```

The following are the main `supervisor.py` commands needed.

| argument            | description                                      | example                                     |
| ------------------- | ------------------------------------------------ | ------------------------------------------- |
| `--init`            | reset all matrices and control variables         | `./supervisor.py 1 --init`                  |
| `--reset`           | reset control variables online                   | `./supervisor.py 1 --reset`                 |
| `--recompute`       | Do poke test, meas imat, compute cmat            | `./supervisor.py 1 --recompute`             |
| `--poke`            | specify poke during poke test                    | `./supervisor.py 1 --recompute --poke 0.01` |
| `--nmodes`          | specify number of modes for controller to act on | `./supervisor.py 1 --recompute --nmodes 50` |
| `--gain` & `--leak` | set leaky integrator gain and leak               | `./supervisor.py 1 --gain 0.3 --leak 0.99`  |
| `--open`            | open the loop immediately                        | `./supervisor.py 1 --open`                  |
| `--close`           | close the loop (using previously set gain/leak)  | `./supervisor.py 1 --close`                 |
| `--status`          | check the status of some variables (WIP)         | `./supervisor.py 1 --status`                |

## Installation

In the `dcs` root directory (one level up from here), run:

```bash
cmake -B build
cmake --build build
```

Note that these instructions can be found in [dcs/README.md](../README.md)

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
| `flux_mask`         |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `strehl_mask`       |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
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
