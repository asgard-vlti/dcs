# dcs
Detector Control System for Heimdallr/Baldr

## Subsystems
This repository covers multiple control modules, all within the Detector Control System. Modules with a `README.md` are hyperlinked.
 - Communication with wag and the ESO systems, which sees everything as "DCS"
    -  `back_end_server`
    -  `mcs_client`
 - C-RED1
    - `asgard-cred1-server`
 - DM Interface
    - `asgard-dm-server`
 - Baldr, which has 2 fundamentally different modes (image-plane and Zernike WFS).
    - [`baldr_jcr`](./baldr_jcr/README.md)
    - [`baldr_python_rtc`](./baldr_python_rtc/README.md)
    - `baldr_tt`
    - `minimal_baldr_python_rtc`
    - `pyeng_baldr`
 - Heimdallr, including real-time and python tools
    - `heimdallr`
    - `pyeng_heimdallr`
 - Inter-process communication
    - [`commander`](./commander/README.md)
    - `libImageStreamIO`
 - Scripts, monitoring and utilities
    - `cred1view`
    - `dcs`
    - `lab-dm-tools`
    - [`simulation`](./simulation/readme.md)
    - `utils`
 - Legacy/Redundant
    - `lib`, contains copy of static libraries which are now built by `cmake`
    - `archived`

## Dependencies
A possibly incomplete list of dependencies is below:
 - `cmake`
 - `nlohmann_json`
 - `boost`
 - `cppzmq`
 - `fmt`
 - `fftw` (including `libfftw3-dev`)
 - `tomlplusplus`
 - `cfitsio`
 - `libb64`

## Installation
The simplest installation of the executables and libraries in this repo is
done using `cmake`. In this directory, run:

```bash
cmake -B build  # set up the build system in ./build directory
cmake --build build  # execute the build, producing outputs in ./build
```

So far, the `cmake` only manages (see `./CMakeLists.txt`):
 - `baldr_jcr` (executable)
 - `ImageStreamIO` (library)
 - `Commander` (library)

For the other targets, use whatever build system those targets provide.

## Beam flattening sources

Run `dcs/cmd_scripts/flatten_beam.py` on `mimir` in the `asg` conda environment.
The default `--source live` acquires a clear pupil before preparing the selected
target. Saved sources can replace this acquisition or supply the final ZWFS
target directly:

```bash
python dcs/cmd_scripts/flatten_beam.py 1 --target amp-model --source saved-pupil
python dcs/cmd_scripts/flatten_beam.py 1 --target amp-model --source saved-ref
```

| Source | Current product | Selected FITS image | Behavior |
| --- | --- | --- | --- |
| `saved-pupil` | `~/etc/b-pupils/beam1.fits` | `CLEAR_PUPIL` | Fit the clear pupil and prepare the selected target |
| `saved-ref` | `~/etc/b-references/beam1.fits` | `PHASE_MASK` | Use the ZWFS image directly, skipping pupil fitting and model generation |

The beam argument selects `beam1`, `beam2`, etc. These fixed filenames represent
the latest products and are maintained externally; the script does not search
for timestamped files or save new input products. FITS inputs follow the named
image convention in `baldr_reference_calibration`.

To use NumPy files, change the module-level `USE_FITS` boolean to `False` in
`flatten_beam.py`. The same paths then end in `.npy`, and each file contains one
32x32 clear-pupil or ZWFS image according to its directory. There is no CLI format
flag or automatic fallback between formats. `.npz` is not supported.

Saved images must already be calibrated, finite, numeric 32x32 arrays with
positive total flux. Invalid or missing inputs fail before hardware setup.
`saved-pupil` supports all targets; `saved-ref` accepts `model` and `amp-model`
with identical behavior and rejects `stddev`.

All sources still take a fresh dark (including the existing BMY movement) and
optimize against live camera frames using the DM. Saved images are not corrected
with this new dark. Optimization, operator confirmation, and night-standard
flat saving/loading retain their existing behavior.

Run the source-loading and mocked hardware checks without instrument access:

```bash
python -m unittest discover -s tests -p 'test_flatten_beam.py' -q
```

## Camera power cycle

On `mimir`, with the `asg` environment active and `asgard-alignment` installed,
run `power-cycle-camera` to recover a C-RED1 that is running but no longer returns
settings. The command shuts down the camera, cycles PDU outlet 6, restarts the
camera server, enables cooling, restarts Kaya, and monitors temperature and camera
status. It waits up to 15 minutes for detector temperature below 81 and camera
CLI status `ready`, then restarts the server again and verifies cropped mode. If
cooling times out, the camera remains powered and the first restarted server
remains running.

The offline sequence checks are:

```bash
python -m unittest discover -s tests -p 'test_power_cycle_camera.py' -q
```

## Static DM sparkles

Run `sparkles` on `mimir` in the `asg` conda environment. The optional beam
argument selects the initial beam (`sparkles 2`); the GUI can select beams 1–4.
Set the angle from the DM actuator grid's horizontal axis, separation in
`lambda/D` (including fractional values from 0.1 to 5.0), and amplitude in
normalized DM command units (0 to 0.2). The amplitude limits the peak of the
combined two-mode pattern. Checking **On** writes a fixed pattern with two
orthogonal Fourier modes, producing four symmetric spots; unchecking clears it.

The utility owns `/dev/shm/dm<beam>disp03.im.shm`. Opening the GUI clears that
channel, replacing any `dm-zernike` or other pattern there. Switching beams
clears the previous beam's channel; closing the GUI normally clears the selected
channel. No other DM channel is changed. The requested angle is in DM grid
coordinates; camera orientation requires per-beam optical calibration.
Run only one sparkles GUI per beam. At exactly 5 `lambda/D`, the DM reaches its
actuator sampling limit; use a smaller distance if the spots overlap or alias.

Offline checks use fake shared-memory streams and an offscreen Qt window:

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -p 'test_sparkles.py' -q
```

On `mimir`, verify the `disp03` and combined maps in `dmview`, and measure spot
separation and orientation on a camera before operational use.
