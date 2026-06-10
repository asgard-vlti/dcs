# dcs
Detector Control System for Heimdallr/Baldr

## Dependencies
If you are using nix for package management, you can install all dependencies
into a `devShell` by calling:
```bash
nix develop
```

If you are not using nix, you can inspect the `packages` listed
in the `./flake.nix` file to  see which libraries you might need to install.

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