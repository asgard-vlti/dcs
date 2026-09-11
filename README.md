# dcs
Detector Control System for Heimdallr/Baldr

## Subsystems
This repository covers multiple control modules. Modules with a `README.md` are hyperlinked.
 - C-RED1
    - `asgard-cred1-server`
 - DM Interface
    - `asgard-dm-server`
 - Baldr
    - `baldr`
    - [`baldr_jcr`](./baldr_jcr/README.md)
    - [`baldr_python_rtc`](./baldr_python_rtc/README.md)
    - `baldr_tt`
    - `minimal_baldr_python_rtc`
 - Inter-process communication
    - [`commander`](./commander/README.md)
    - `libImageStreamIO`
 - Monitoring
    - ?
 - Unknown, To Be Discussed with Mike
    - `back_end_server`
    - `calibration_frames`
    - `catch2`
    - `cred1view`
    - `dcs`
    - `heimdallr`
    - `lab-dm-tools`
    - `mcs_client`
    - `pyeng_baldr`
    - `pyeng_heimdallr`
    - [`simulation`](./simulation/readme.md)
    - `utils`
 - Legacy/Redundant
    - `lib`, contains copy of static libraries which are now built by `cmake`


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
