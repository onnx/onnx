<!--- SPDX-License-Identifier: Apache-2.0 -->

<p align="center"><img width="40%" src="https://github.com/onnx/onnx/raw/master/docs/ONNX_logo_main.png" /></p>

[![Build Status](https://img.shields.io/azure-devops/build/onnx-pipelines/onnx/7?label=Linux&logo=Azure-Pipelines)](https://dev.azure.com/onnx-pipelines/onnx/_build/latest?definitionId=7&branchName=master)
[![Build Status](https://img.shields.io/azure-devops/build/onnx-pipelines/onnx/5?label=Windows&logo=Azure-Pipelines)](https://dev.azure.com/onnx-pipelines/onnx/_build/latest?definitionId=5&branchName=master)
[![Build Status](https://img.shields.io/azure-devops/build/onnx-pipelines/onnx/6?label=MacOS&logo=Azure-Pipelines)](https://dev.azure.com/onnx-pipelines/onnx/_build/latest?definitionId=6&branchName=master)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3313/badge)](https://bestpractices.coreinfrastructure.org/projects/3313)

[Open Neural Network Exchange (ONNX)](https://onnx.ai) is an open ecosystem that empowers AI developers
to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard
data types. Currently we focus on the capabilities needed for inferencing (scoring).

ONNX is [widely supported](http://onnx.ai/supported-tools) and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community. We invite the community to join us and further evolve ONNX.

# Use ONNX
* [Tutorials for creating ONNX models](https://github.com/onnx/tutorials).
* [Pre-trained ONNX models](https://github.com/onnx/models)

# Learn about the ONNX spec
* [Overview][overview]
* [ONNX intermediate representation spec][ir]
* [Versioning principles of the spec][versioning]
* [Operators documentation][operators]
* [Python API Overview][python_api]

# Programming utilities for working with ONNX Graphs
* [Shape and Type Inference][shape_inference]
* [Graph Optimization](https://github.com/onnx/optimizer)
* [Opset Version Conversion][version_converter]

# Contribute
ONNX is a [community project][community]. We encourage you to join the effort and contribute feedback, ideas, and code. You can participate in the [SIGs][sigs] and [Working Groups][wgs] to shape the future of ONNX.

Check out our [contribution guide][contributing] to get started.

If you think some operator should be added to ONNX specification, please read
[this document][new_op].

# Discuss
We encourage you to open [Issues](https://github.com/onnx/onnx/issues), or use [Slack](https://slack.lfai.foundation/) for more real-time discussion

# Follow Us
Stay up to date with the latest ONNX news. [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter](https://twitter.com/onnxai)]






# Installation

## Prerequisites

```
numpy >= 1.16.6
protobuf >= 3.12.2
six
typing-extensions >= 3.6.2.1
```

## Official Python packages
ONNX released packages are published in PyPi.
```
pip install numpy protobuf==3.16.0
pip install onnx
```

[Weekly packages](https://test.pypi.org/project/onnx-weekly/) are published in test pypi to enable experimentation and early testing.


## Conda packages
A binary build of ONNX is available from [Conda](https://conda.io), in [conda-forge](https://conda-forge.org/):
```
conda install -c conda-forge numpy protobuf==3.16.0 libprotobuf=3.16.0
conda install -c conda-forge onnx
```

You can also use the [onnx-dev docker image](https://hub.docker.com/r/onnx/onnx-dev) for a Linux-based installation without having to worry about dependency versioning.


## Build ONNX from Source
Before building from source uninstall any existing versions of onnx `pip uninstall onnx`.  

Generally spreaking, you need to install [protobuf C/C++ libraires and tools](https://github.com/protocolbuffers/protobuf) before proceeding forward. Then depending on how you installed protobuf, you need to set environment variable CMAKE_ARGS to "-DONNX_USE_PROTOBUF_SHARED_LIBS=ON" or "-DONNX_USE_PROTOBUF_SHARED_LIBS=OFF".  For example, you may need to run the following command:

Linux:
```bash
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
```
Windows:
```bat
set CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
```

The ON/OFF depends on what kind of protobuf library you have. Shared libraries are files ending with \*.dll/\*.so/\*.dylib. Static libraries are files ending with \*.a/\*.lib. This option depends on how you get your protobuf library and how it was built. And it is default OFF. You don't need to run the commands above if you'd prefer to use a static protobuf library.


### Windows
If you are building ONNX from source, it is recommended that you also build Protobuf locally as a static library. The version distributed with conda-forge is a DLL, but ONNX expects it to be a static library. Building protobuf locally also lets you control the verison of protobuf. The tested and recommended version is 3.16.0.

The instructions in this README assume you are using Visual Studio.  It is recommended that you run all the commands from a shell started from "x64 Native Tools Command Prompt for VS 2019" and keep the build system generator for cmake (e.g., cmake -G "Visual Studio 16 2019") consistent while building protobuf as well as ONNX.

You can get protobuf by running the following commands:
```bat
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.16.0
cd cmake
cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX=<protobug_install_dir> -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF .
msbuild protobuf.sln /m /p:Configuration=Release
msbuild INSTALL.vcxproj /p:Configuration=Release
```
Then it will be built as a static library and installed to <protobug_install_dir>. Please add the bin directory(which contains protoc.exe) to your PATH.

```bat
set PATH=<protobug_install_dir>/bin;%PATH%
```

Please note: if your protobug_install_dir contains spaces, **do not** add quotation marks around it. 

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
pip install -e .
```
### Linux

First, you need to install protobuf. 

Ubuntu users: the quickest way to install protobuf is to run

```bash
apt-get install python3-pip python3-dev libprotobuf-dev protobuf-compiler
```

Then you can build ONNX as:
```
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
git clone --recursive https://github.com/onnx/onnx.git
cd onnx
# prefer lite proto
set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
pip install -e .
```

Otherwise, you may need to install it from source. You can use the following commands to do it:

Debian/Ubuntu:
```
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.16.0
git submodule update --init --recursive
mkdir build_source && cd build_source
cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install
```

CentOS/RHEL/Fedora:
```
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.16.0
git submodule update --init --recursive
mkdir build_source && cd build_source
cmake ../cmake  -DCMAKE_INSTALL_LIBDIR=lib64 -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install
```

Here "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" is crucial. By default static libraries are built without "-fPIC" flag, they are not position independent code. But shared libraries must be position independent code. Python C/C++ extensions(like ONNX) are shared libraries. So if a static library was not built with "-fPIC", it can't be linked to such a shared library.

Once build is successful, update PATH to include protobuf paths.

Then you can build ONNX as:
```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
# prefer lite proto
set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
pip install -e .
```

* **Mac**
```
export NUM_CORES=`sysctl -n hw.ncpu`
brew update
brew install autoconf && brew install automake
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.16.0/protobuf-cpp-3.16.0.tar.gz
tar -xvf protobuf-cpp-3.16.0.tar.gz
cd protobuf-3.16.0
mkdir build_source && cd build_source
cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j${NUM_CORES}
make install
```

Once build is successful, update PATH to include protobuf paths.

Then you can build ONNX as:
```
git clone --recursive https://github.com/onnx/onnx.git
cd onnx
# prefer lite proto
set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
pip install -e .
```


## Verify Installation
After installation, run

```
python -c "import onnx"
```

to verify it works.


## Common Build Options
For full list refer to CMakeLists.txt  
**Environment variables**  
* `USE_MSVC_STATIC_RUNTIME` should be 1 or 0, not ON or OFF. When set to 1 onnx links statically to runtime library.  
**Default**: USE_MSVC_STATIC_RUNTIME=0

* `DEBUG` should be 0 or 1. When set to 1 onnx is built in debug mode. or debug versions of the dependencies, you need to open the [CMakeLists file][CMakeLists] and append a letter `d` at the end of the package name lines. For example, `NAMES protobuf-lite` would become `NAMES protobuf-lited`.  
**Default**: Debug=0

**CMake variables**
* `ONNX_USE_PROTOBUF_SHARED_LIBS` should be ON or OFF.  
**Default**: ONNX_USE_PROTOBUF_SHARED_LIBS=OFF USE_MSVC_STATIC_RUNTIME=0
`ONNX_USE_PROTOBUF_SHARED_LIBS` determines how onnx links to protobuf libraries.  
    - When set to ON - onnx will dynamically link to protobuf shared libs, PROTOBUF_USE_DLLS will be defined as described [here](https://github.com/protocolbuffers/protobuf/blob/master/cmake/README.md#dlls-vs-static-linking), Protobuf_USE_STATIC_LIBS will be set to OFF and `USE_MSVC_STATIC_RUNTIME` must be 0.  
    - When set to OFF - onnx will link statically to protobuf, and Protobuf_USE_STATIC_LIBS will be set to ON (to force the use of the static libraries) and `USE_MSVC_STATIC_RUNTIME` can be 0 or 1.  

* `ONNX_USE_LITE_PROTO` should be ON or OFF. When set to ON onnx uses lite protobuf instead of full protobuf.  
**Default**: ONNX_USE_LITE_PROTO=OFF

* `ONNX_WERROR` should be ON or OFF. When set to ON warnings are treated as errors.  
**Default**: ONNX_WERROR=OFF in local builds, ON in CI and release pipelines.


## Common Errors
* Note: the `import onnx` command does not work from the source checkout directory; in this case you'll see `ModuleNotFoundError: No module named 'onnx.onnx_cpp2py_export'`. Change into another directory to fix this error.

* Building ONNX on Ubuntu works well, but on CentOS/RHEL and other ManyLinux systems, you might need to open the [CMakeLists file][CMakeLists] and replace all instances of `/lib` with `/lib64`.

# Testing

ONNX uses [pytest](https://docs.pytest.org) as test driver. In order to run tests, you will first need to install pytest:

```
pip install pytest nbval
```

After installing pytest, use the following command to run tests.

```
pytest
```

# Development

Check out the [contributor guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) for instructions.

# License

[Apache License v2.0][license]

# Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)


<!-- links
NOTE: these are absolute rather than repo-relative because we
this document is rendered by PyPI which doesn't seem to handle
relative links properly. See https://github.com/dhimmel/obonet/issues/12.
-->
[overview]: https://github.com/onnx/onnx/blob/master/docs/Overview.md
[ir]: https://github.com/onnx/onnx/blob/master/docs/IR.md
[versioning]: https://github.com/onnx/onnx/blob/master/docs/Versioning.md
[operators]: https://github.com/onnx/onnx/blob/master/docs/Operators.md
[python_api]: https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md
[shape_inference]: https://github.com/onnx/onnx/blob/master/docs/ShapeInference.md
[version_converter]: https://github.com/onnx/onnx/blob/master/docs/VersionConverter.md
[new_op]: https://github.com/onnx/onnx/blob/master/docs/AddNewOp.md
[community]: https://github.com/onnx/onnx/tree/master/community
[sigs]: https://github.com/onnx/onnx/tree/master/community/sigs.md
[wgs]: https://github.com/onnx/onnx/tree/master/community/working-groups.md
[contributing]: https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md
[CMakeLists]: https://github.com/onnx/onnx/blob/master/CMakeLists.txt
[license]: https://github.com/onnx/onnx/blob/master/LICENSE
