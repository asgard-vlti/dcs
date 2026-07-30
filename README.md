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
 - `fftw`
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