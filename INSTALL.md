<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Installation

## Official Python packages

ONNX released packages are published in PyPi.

```sh
pip install onnx  # or pip install onnx[reference] for optional reference implementation dependencies
```

[ONNX weekly packages](https://pypi.org/project/onnx-weekly/) are published in PyPI to enable experimentation and early testing.

## vcpkg packages

ONNX is in the maintenance list of [vcpkg](https://github.com/microsoft/vcpkg), you can easily use vcpkg to build and install it.

```sh
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat # For powershell
./bootstrap-vcpkg.sh # For bash
./vcpkg install onnx
```

## Conda packages

A binary build of ONNX is available from [Conda](https://conda.io), in [conda-forge](https://conda-forge.org/):

```sh
conda install -c conda-forge onnx
```

## Build ONNX from Source

Before building from source uninstall any existing versions of ONNX `pip uninstall onnx`.

C++17 or higher C++ compiler version is required to build ONNX from source. Still, users can specify their own `CMAKE_CXX_STANDARD` version for building ONNX.

If you don't have protobuf installed, ONNX will internally download and build protobuf for ONNX build.

Or, you can manually install [protobuf C/C++ libraries and tools](https://github.com/protocolbuffers/protobuf) with specified version before proceeding forward. Then depending on how you installed protobuf, you need to set environment variable CMAKE_ARGS to "-DONNX_USE_PROTOBUF_SHARED_LIBS=ON" or "-DONNX_USE_PROTOBUF_SHARED_LIBS=OFF".  For example, you may need to run the following command:

Linux:

```sh
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
```

Windows:

```bat
set CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
```

The ON/OFF depends on what kind of protobuf library you have. Shared libraries are files ending with \*.dll/\*.so/\*.dylib. Static libraries are files ending with \*.a/\*.lib. This option depends on how you get your protobuf library and how it was built. And it is default OFF. You don't need to run the commands above if you'd prefer to use a static protobuf library.

### Windows

If you are building ONNX from source, it is recommended that you also build Protobuf locally as a static library. The version distributed with conda-forge is a DLL, but ONNX expects it to be a static library. Building protobuf locally also lets you control the version of protobuf. The tested and recommended version is 3.21.12.

The instructions in this README assume you are using Visual Studio.  It is recommended that you run all the commands from a shell started from "x64 Native Tools Command Prompt for VS 2019" and keep the build system generator for cmake (e.g., cmake -G "Visual Studio 16 2019") consistent while building protobuf as well as ONNX.

You can get protobuf by running the following commands:

```bat
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v21.12
cd cmake
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=<protobuf_install_dir> -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF .
msbuild protobuf.sln /m /p:Configuration=Release
msbuild INSTALL.vcxproj /p:Configuration=Release
```

Then it will be built as a static library and installed to <protobuf_install_dir>. Please add the bin directory(which contains protoc.exe) to your PATH.

```bat
set CMAKE_PREFIX_PATH=<protobuf_install_dir>;%CMAKE_PREFIX_PATH%
```

Please note: if your protobuf_install_dir contains spaces, **do not** add quotation marks around it.

Alternative: if you don't want to change your PATH, you can set ONNX_PROTOC_EXECUTABLE instead.

```bat
set CMAKE_ARGS=-DONNX_PROTOC_EXECUTABLE=<full_path_to_protoc.exe>
```

Then you can build ONNX as:

```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
# prefer lite proto
set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
pip install -e . -v
```

### Linux

First, you need to install protobuf. The minimum Protobuf compiler (protoc) version required by ONNX is 3.6.1. Please note that old protoc versions might not work with `CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON`.

Ubuntu 20.04 (and newer) users may choose to install protobuf via

```sh
apt-get install python3-pip python3-dev libprotobuf-dev protobuf-compiler
```

In this case, it is required to add `-DONNX_USE_PROTOBUF_SHARED_LIBS=ON` to CMAKE_ARGS in the ONNX build step.

A more general way is to build and install it from source. See the instructions below for more details.

<details>
  <summary> Installing Protobuf from source </summary>

  Debian/Ubuntu:

  ```sh
    git clone https://github.com/protocolbuffers/protobuf.git
    cd protobuf
    git checkout v21.12
    git submodule update --init --recursive
    mkdir build_source && cd build_source
    cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    make install
  ```

  CentOS/RHEL/Fedora:

  ```sh
    git clone https://github.com/protocolbuffers/protobuf.git
    cd protobuf
    git checkout v21.12
    git submodule update --init --recursive
    mkdir build_source && cd build_source
    cmake ../cmake  -DCMAKE_INSTALL_LIBDIR=lib64 -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    make install
  ```

  Here "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" is crucial. By default static libraries are built without "-fPIC" flag, they are not position independent code. But shared libraries must be position independent code. Python C/C++ extensions(like ONNX) are shared libraries. So if a static library was not built with "-fPIC", it can't be linked to such a shared library.

  Once build is successful, update PATH to include protobuf paths.

</details>

Then you can build ONNX as:

```sh
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
# Optional: prefer lite proto
export CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
pip install -e . -v
```

### Mac

```sh
export NUM_CORES=`sysctl -n hw.ncpu`
brew update
brew install autoconf && brew install automake
wget https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protobuf-cpp-3.21.12.tar.gz
tar -xvf protobuf-cpp-3.21.12.tar.gz
cd protobuf-3.21.12
mkdir build_source && cd build_source
cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j${NUM_CORES}
make install
```

Once build is successful, update PATH to include protobuf paths.

Then you can build ONNX as:

```sh
git clone --recursive https://github.com/onnx/onnx.git
cd onnx
# Optional: prefer lite proto
set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
pip install -e . -v
```

## Verify Installation

After installation, run

```sh
python -c "import onnx"
```

to verify it works.

## Common Build Options

For full list refer to CMakeLists.txt

### Environment variables

* `USE_MSVC_STATIC_RUNTIME` should be 1 or 0, not ON or OFF. When set to 1 ONNX links statically to runtime library.
**Default**: `USE_MSVC_STATIC_RUNTIME=0`

* `DEBUG` should be 0 or 1. When set to 1 ONNX is built in debug mode. or debug versions of the dependencies, you need to open the [CMakeLists file](https://github.com/onnx/onnx/blob/main/CMakeLists.txt) and append a letter `d` at the end of the package name lines. For example, `NAMES protobuf-lite` would become `NAMES protobuf-lited`.
**Default**: `Debug=0`

### CMake variables

* `ONNX_USE_PROTOBUF_SHARED_LIBS` should be `ON` or `OFF`.
**Default**: `ONNX_USE_PROTOBUF_SHARED_LIBS=OFF USE_MSVC_STATIC_RUNTIME=0`
`ONNX_USE_PROTOBUF_SHARED_LIBS` determines how ONNX links to protobuf libraries.
  * When set to `ON` - ONNX will dynamically link to protobuf shared libs, PROTOBUF_USE_DLLS will be defined as described [here](https://github.com/protocolbuffers/protobuf/blob/main/cmake/README.md#dlls-vs-static-linking).
  * When set to `OFF` - ONNX will link statically to protobuf.

* `ONNX_USE_LITE_PROTO` should be `ON` or `OFF`. When set to `ON` ONNX uses lite protobuf instead of full protobuf.
**Default**: `ONNX_USE_LITE_PROTO=OFF`

* `ONNX_WERROR` should be `ON` or `OFF`. When set to `ON` warnings are treated as errors.
**Default**: `ONNX_WERROR=OFF` in local builds, `ON` in CI and release pipelines.

## Common Errors

* Note: the `import onnx` command does not work from the source checkout directory; in this case you'll see `ModuleNotFoundError: No module named 'onnx.onnx_cpp2py_export'`. Change into another directory to fix this error.

* If you run into any issues while building Protobuf as a static library, please ensure that shared Protobuf libraries, like libprotobuf, are not installed on your device or in the conda environment. If these shared libraries exist, either remove them to build Protobuf from source as a static library, or skip the Protobuf build from source to use the shared version directly.

* If you run into any issues while building ONNX from source, and your error message reads, `Could not find pythonXX.lib`, ensure that you have consistent Python versions for common commands, such as `python` and `pip`. Clean all existing build files and rebuild ONNX again.
