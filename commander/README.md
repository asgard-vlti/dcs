# Commander

Commander is a command line tool interface library, based on the Pyxis project's
commander written by Julien Tom Bernard, which in turn was based on earlier ZMQ
work by Mike Ireland, and inspired by SUSI and CHARA code. The intent in this implementation
is to limit dependencies and maximise simplicity. 

## Quick start

Let's start by creating command bindings for an extremely simple function, which
adds two numbers and returns their result:

```c++
int add(int i, int j) {
    return i + j;
}
```

For simplicity we'll put both this function and the binding code into
a file named `example.cpp` with the following contents:

```cpp
#include <commander/commander.h>

int add(int i, int j) {
    return i + j;
}

COMMANDER_REGISTER(m) {
    m.def(
        "add",
        &add,
        "A function that adds two numbers",
        commander::arg("i", "First integer to add."),
        commander::arg("j", "Second integer to add.")
    );
}
```

`commander::arg` supplies the argument name and description while Commander
continues to infer its type from the callable. A third parameter supplies a
default value, for example `commander::arg("scale", "Scale factor.", 1.0)`.
The older `"name"_arg` syntax remains supported without a description.

Now, all you need is to compile the source file including the library flag -lcommander, including
the lib directory of this repository in the -L flag and run the executable with for instance the
`commander::Server`:

```cpp

int main(int argc, char* argv[]) {
    commander::Server(argc, argv).run();
}
```

and use it as follows:

```bash
./example --command add [2,3]
5
```

To run as a socket, you have to give the full address of the socket port, e.g.:

```bash
./example --socket tcp://127.0.0.1:3000
```

## Installation

### Dependencies and Deployment:

`commander` requires a C++ compiler compatible with C++11, and the 4 libraries: nlohmann_json/3.10.5, 
cppzmq/4.8.1, boost/1.78.0 and fmt/8.1.1. These can be installed as Ubuntu packages using:

```bash
sudo apt-get install nlohmann-json3-dev libfmt-dev libzmq3-dev libboost-all-dev
``` 

To complile the library in the src directory, just type:

```bash
make
```

However, if you don't want to hack the Makefile for a new system, then you also need 
to help us install autoconf and autotools. In this case, we'll want something like:

```bash
./configure
make
sudo make install
```

## Build documentation

To build the documentation, you need to install `doxygen`, `breathe`, `Sphinx`, `sphinx-rtd-theme`, `sphinx-rtd-dark-mode`.

To install doxygen, refer to your OS package manager. For the rest, run:

```bash
pip install -r requirements.txt
```

To build the documentation, run:

```bash
cd docs
make html
```

It will produce the documentation in `build/docs/sphinx/html`.
