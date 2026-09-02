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

### Building ONNX from source in a conda-forge-based environment

A conda-forge-based development environment is provided through the cross-platform [Pixi package manager](https://prefix.dev/).
The `pixi.toml` file in the root directory defines various tasks for common development workflows.
Running
```sh
pixi run install
```
builds the C++ component and installs it as an editable Python package.

After the installation has completed one can run the gtest and pytest suites via the pixi-tasks of the same name:

```sh
pixi run gtest
```

and

```sh
pixi run pytest
```


## Build ONNX from Source with manually managed dependencies

Before building from source uninstall any existing versions of ONNX via `pip uninstall onnx`.

C++17 or higher C++ compiler version is required to build ONNX from source. Still, users can specify their own `CMAKE_CXX_STANDARD` version for building ONNX.

Protobuf is required for ONNX. If you don't have Protobuf installed, ONNX will internally download and build the version pinned in [`sbom.cdx.json`](sbom.cdx.json).

Protobuf uses different major-version numbers for some language runtimes. In this checkout, the Python package requirement is `protobuf 6.31.1` in [`pyproject.toml`](pyproject.toml), while the corresponding upstream C++ library and compiler release is `v31.1` in [`sbom.cdx.json`](sbom.cdx.json). These version strings are related but are not interchangeable: use the `v31.1` tag when building the C++ library or `protoc` from source. See [Protobuf version support](https://protobuf.dev/support/version-support/) for an explanation of the versioning scheme.

Or, you can manually install [Protobuf C/C++ libraries and tools](https://github.com/protocolbuffers/protobuf) with specified version before proceeding forward. Then depending on how you installed Protobuf, you need to set environment variable CMAKE_ARGS to "-DONNX_USE_PROTOBUF_SHARED_LIBS=ON" or "-DONNX_USE_PROTOBUF_SHARED_LIBS=OFF". For example, you may need to run the following command:

Linux or Mac:

```sh
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
```

Windows:

```bat
set "CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
```

The ON/OFF depends on what kind of Protobuf library you have. Shared libraries are files ending with \*.dll/\*.so/\*.dylib. Static libraries are files ending with \*.a/\*.lib. This option depends on how you get your Protobuf library and how it was built. Because its default value is OFF, you don't need to run the commands above if you'd prefer to use a static Protobuf library.

### Windows

```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
# prefer lite proto
set "CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON -DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
pip install -e . -v
```


#### Building Protobuf from source

Normally, ONNX's CMake configuration downloads the pinned Protobuf release automatically. If you need to provide an external Protobuf installation, build the C++ release recorded in [`sbom.cdx.json`](sbom.cdx.json), currently `v31.1`. Build it as either a static or shared library and set `ONNX_USE_PROTOBUF_SHARED_LIBS` consistently.

Run the following commands from the x64 Native Tools Command Prompt for Visual Studio 2022. Keep the CMake generator consistent while building Protobuf and ONNX.

You can build Protobuf from source by running the following commands:

```bat
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v31.1
git submodule update --init --recursive
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_INSTALL_PREFIX=<protobuf_install_dir> -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF
cmake --build . --config Release --target install
```

Then it will be built as a static library and installed to <protobuf_install_dir>. Please add the bin directory(which contains protoc.exe) to your PATH.

```bat
set CMAKE_PREFIX_PATH=<protobuf_install_dir>;%CMAKE_PREFIX_PATH%
```

Please note: if your protobuf_install_dir contains spaces, **do not** add quotation marks around it.

Alternative: if you have local Protobuf executable and want to use it for ONNX, you can set ONNX_PROTOC_EXECUTABLE instead.

```bat
set "CMAKE_ARGS=-DONNX_PROTOC_EXECUTABLE=<full_path_to_protoc.exe>"
```

Then you can build ONNX as:

```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
# prefer lite proto
set "CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON"
pip install -e . -v
```

### Linux

ONNX can use an external Protobuf installation. The C++ library and `protoc` release pinned and tested by this checkout is `v31.1`, as recorded in [`sbom.cdx.json`](sbom.cdx.json). This is distinct from the Python package version `6.31.1`.

Ubuntu users may install Protobuf using the system package manager:

```sh
apt-get install python3-pip python3-dev libprotobuf-dev protobuf-compiler
```
In this case, ONNX can detect and use the system Protobuf installation. Users of other Linux distributions can install the Protobuf libraries similarly. Distribution versions can differ from the version tested by ONNX; for a predictable build, let ONNX download its pinned dependency or build that release from source.

To build and install the pinned Protobuf release from source, use the instructions below.

<details>
  <summary> Installing Protobuf from source </summary>

```sh
  git clone https://github.com/protocolbuffers/protobuf.git
  cd protobuf
  git checkout v31.1
  git submodule update --init --recursive
  mkdir build_source && cd build_source
  cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
  cmake --build . --target install
```

  Here "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" is crucial. By default static libraries are built without "-fPIC" flag, they are not position independent code. But shared libraries must be position independent code. Python C/C++ extensions(like ONNX) are shared libraries. So if a static library was not built with "-fPIC", it can't be linked to such a shared library.

  Once build is successful, update PATH to include Protobuf paths so that ONNX can find Protobuf.

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
brew update
brew install cmake
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v31.1
git submodule update --init --recursive
mkdir build_source && cd build_source
cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
cmake --build . --target install
```

Once build is successful, update PATH to include Protobuf paths so that ONNX can find Protobuf.

Then you can build ONNX as:

```sh
git clone --recursive https://github.com/onnx/onnx.git
cd onnx
# Optional: prefer lite proto
export CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
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

* `DEBUG` should be 0 or 1. When set to 1 ONNX is built in debug mode. For debug versions of the dependencies, you need to open the [CMakeLists file](https://github.com/onnx/onnx/blob/main/CMakeLists.txt) and append a letter `d` at the end of the package name lines. For example, `NAMES protobuf-lite` would become `NAMES protobuf-lited`.
**Default**: `Debug=0`

### CMake variables

* `ONNX_USE_PROTOBUF_SHARED_LIBS` should be `ON` or `OFF`.
**Default**: `ONNX_USE_PROTOBUF_SHARED_LIBS=OFF USE_MSVC_STATIC_RUNTIME=0`
`ONNX_USE_PROTOBUF_SHARED_LIBS` determines how ONNX links to Protobuf libraries.
  * When set to `ON` - ONNX will dynamically link to Protobuf shared libs, PROTOBUF_USE_DLLS will be defined as described [here](https://github.com/protocolbuffers/protobuf/blob/main/cmake/README.md#dlls-vs-static-linking).
  * When set to `OFF` - ONNX will link statically to Protobuf.

* `ONNX_USE_LITE_PROTO` should be `ON` or `OFF`. When set to `ON` ONNX uses lite Protobuf instead of full Protobuf.
**Default**: `ONNX_USE_LITE_PROTO=OFF`

* `ONNX_WERROR` should be `ON` or `OFF`. When set to `ON` warnings are treated as errors.
**Default**: `ONNX_WERROR=OFF` in local builds, `ON` in CI and release pipelines.

* `nanobind_DIR` can be set to the directory that contains `nanobindConfig.cmake` (for example,
  `python -m nanobind --cmake_dir`) if CMake cannot find nanobind. You can also set
  `CMAKE_PREFIX_PATH` instead.

* `FETCHCONTENT_FULLY_DISCONNECTED` is intended for subsequent re-configures after
  dependencies are already populated. It does not prevent network access on the initial
  configure; for fully offline first-run builds, prefer a
  [dependency provider](https://cmake.org/cmake/help/latest/module/FetchContent.html#dependency-providers)
  or provide dependencies locally (for example, via `nanobind_DIR` or `CMAKE_PREFIX_PATH`).

## Common Errors

* Note: the `import onnx` command does not work from the source checkout directory; in this case you'll see `ModuleNotFoundError: No module named 'onnx.onnx_cpp2py_export'`. Change into another directory to fix this error.

* If you run into any issues while building Protobuf as a static library, please ensure that shared Protobuf libraries, like libprotobuf, are not installed on your device or in the conda environment. If these shared libraries exist, either remove them to build Protobuf from source as a static library, or skip the Protobuf build from source to use the shared version directly.

* If you run into any issues while building ONNX from source, and your error message reads, `Could not find pythonXX.lib`, ensure that you have consistent Python versions for common commands, such as `python` and `pip`. Clean all existing build files and rebuild ONNX again.
