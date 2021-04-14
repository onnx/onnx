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

## Binaries

A binary build of ONNX is available from [Conda](https://conda.io), in [conda-forge](https://conda-forge.org/):

```
conda install -c conda-forge onnx
```

## Source

If you have installed onnx on your machine, please `pip uninstall onnx` first before the following process of build from source. 

### Linux and MacOS
You will need an install of Protobuf and NumPy to build ONNX.  One easy
way to get these dependencies is via
[Anaconda](https://www.anaconda.com/download/):

```
# Use conda-forge protobuf, as default doesn't come with protoc
conda install -c conda-forge protobuf numpy
```

You can then install ONNX from PyPi (Note: Set environment variable `ONNX_ML=1` for onnx-ml):

```
pip install onnx
```

Alternatively, you can also build and install ONNX locally from source code:

```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
python setup.py install
```

Note: When installing in a non-Anaconda environment, make sure to install the Protobuf compiler before running the pip installation of onnx. For example, on Ubuntu:

```
sudo apt-get install protobuf-compiler libprotoc-dev
pip install onnx
```

### Windows
If you are building ONNX from source on Windows, it is recommended that you also build Protobuf locally as a static library. The version distributed with conda-forge is a DLL and this is a conflict as ONNX expects it to be a static library.

Note that the instructions in this README assume you are using Visual Studio. It is recommended that you run all the commands from a shell started from "Developer Command Prompt for VS 2019" and keep the build system generator for cmake (e.g., cmake -G "Visual Studio 16 2019") consistent.

#### Build Protobuf and ONNX on Windows
Step 1: Build Protobuf locally
```
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout 3.11.x
cd cmake
# Explicitly set -Dprotobuf_MSVC_STATIC_RUNTIME=OFF to make sure protobuf does not statically link to runtime library
cmake -G -A -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=<protobuf_install_dir>
# For example:
# cmake -G "Visual Studio 16 2019" -A x64 -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=..\install
msbuild protobuf.sln /m /p:Configuration=Release
msbuild INSTALL.vcxproj /p:Configuration=Release
```

Step 2: Build ONNX
```
# Get ONNX
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive

# Set environment variables to find protobuf and turn off static linking of ONNX to runtime library.
# Even better option is to add it to user\system PATH so this step can be performed only once.
# For more details check https://docs.microsoft.com/en-us/cpp/build/reference/md-mt-ld-use-run-time-library?view=vs-2017
set PATH=<protobuf_install_dir>\bin;<protobuf_install_dir>\include;<protobuf_install_dir>\libs;%PATH%
set USE_MSVC_STATIC_RUNTIME=0

# use the static installed protobuf
set CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DProtobuf_USE_STATIC_LIBS=ON

# Optional: Set environment variable `ONNX_ML=1` for onnx-ml

# Build ONNX
python setup.py install
```

If you would prefer to use Protobuf from conda-forge instead of building Protobuf from source, you can use the following instructions.

#### Build ONNX on Windows with Anaconda

```
# Use conda-forge protobuf
conda install -c conda-forge numpy libprotobuf=3.11.3 protobuf

# Get ONNX
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive

# Set environment variable for ONNX to use protobuf shared lib
set USE_MSVC_STATIC_RUNTIME=0
set CMAKE_ARGS=-DONNX_USE_PROTOBUF_SHARED_LIBS=ON -DProtobuf_USE_STATIC_LIBS=OFF -DONNX_USE_LITE_PROTO=ON

# Build ONNX
# Optional: Set environment variable `ONNX_ML=1` for onnx-ml

python setup.py install
```

#### Build ONNX on ARM 64
If you are building ONNX on an ARM 64 device, please make sure to install the dependencies appropriately.

```
pip install cython protobuf numpy
sudo apt-get install libprotobuf-dev protobuf-compiler
pip install onnx
```

## Verify Installation
After installation, run

```
python -c "import onnx"
```

to verify it works.


#### Common Errors
**Environment variables**: `USE_MSVC_STATIC_RUNTIME` (should be 1 or 0, not ON or OFF)

**CMake variables**: `ONNX_USE_PROTOBUF_SHARED_LIBS`, `Protobuf_USE_STATIC_LIBS`

If `ONNX_USE_PROTOBUF_SHARED_LIBS` is ON then `Protobuf_USE_STATIC_LIBS` must be OFF and `USE_MSVC_STATIC_RUNTIME` must be 0.
If `ONNX_USE_PROTOBUF_SHARED_LIBS` is OFF then `Protobuf_USE_STATIC_LIBS` must be ON and `USE_MSVC_STATIC_RUNTIME` can be 1 or 0.

Note that the `import onnx` command does not work from the source checkout directory; in this case you'll see `ModuleNotFoundError: No module named 'onnx.onnx_cpp2py_export'`. Change into another directory to fix this error.

Building ONNX on Ubuntu works well, but on CentOS/RHEL and other ManyLinux systems, you might need to open the [CMakeLists file][CMakeLists] and replace all instances of `/lib` with `/lib64`.

If you want to build ONNX on Debug mode, remember to set the environment variable `DEBUG=1`. For debug versions of the dependencies, you need to open the [CMakeLists file][CMakeLists] and append a letter `d` at the end of the package name lines. For example, `NAMES protobuf-lite` would become `NAMES protobuf-lited`.

You can also use the [onnx-dev docker image](https://hub.docker.com/r/onnx/onnx-dev) for a Linux-based installation without having to worry about dependency versioning.

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


<!-- links -->
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
