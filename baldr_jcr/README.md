# baldr RTC

Jesse's implementation of the baldr RTC. 

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
./build/baldr_jcr/baldr ./baldr_jcr/def1.toml --socket=tcp://localhost:17474
```

With the RTC running, you can interact with it via the supervisor. To see a list
of available supervisor commands, run:
```bash
python supervisor.py --help
```

For example, to recompute the interaction and control matrices for a POLC controller
with a gain of 0.3 and leak of 0.999, run:
```bash
python supervisor.py 1 --polc --recompute --gain=0.3 --leak=0.999
```

## Todo:

- [ ] URGENT: refactor to allow one RTC instance per beam, currently there will be naming
collisions.

## RTC Logic

![rtc diagram](./baldr_control_logic.svg)




## Internal Compliance

### Data Object Implementation

| Data Object        | Diagram | c-header | servo loop | baldr | config | Commander | supervisor |
| ------------------ | :-----: | :------: | :--------: | :---: | :----: | :-------: | :--------: |
| `meas_offset`      |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `meas_to_mode`     |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `filter_coeff_in`  |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `filter_coeff_out` |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `mode_offset`      |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `mode_max`         |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `mode_min`         |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `mode_to_com`      |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `com_max`          |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `com_min`          |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `com_dist_buffer`  |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |
| `com_offset`       |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :x:     |
| `delay`            |  :ok:   |   :ok:   |    :ok:    |  :x:  |  :ok:  |    :x:    |    :x:     |
| `com_to_meas`      |  :ok:   |   :ok:   |    :ok:    | :ok:  |  :ok:  |   :ok:    |    :ok:    |

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
| `remove_offset`     |    :ok:     |  :x:   |
| `inject_dm_signal`  |    :ok:     |  :x:   |
