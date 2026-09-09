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
