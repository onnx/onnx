
<p align="center"><img width="40%" src="docs/ONNX_logo_main.png" /></p>

[![Build Status](https://img.shields.io/travis/onnx/onnx/master.svg?label=Linux)](https://travis-ci.org/onnx/onnx)
[![Build status](https://img.shields.io/appveyor/ci/onnx/onnx/master.svg?label=Windows)](https://ci.appveyor.com/project/onnx/onnx)
[![Build Status](https://img.shields.io/jenkins/s/http/powerci.osuosl.org/onnx-ppc64le-nightly-build.svg?label=Linux%20ppc64le)](http://powerci.osuosl.org/job/onnx-ppc64le-nightly-build/)

[Open Neural Network Exchange (ONNX)](https://onnx.ai) is an open ecosystem that empowers AI developers
to choose the right tools as their project evolves. ONNX provides an open source format for AI models, both deep learning and traditional ML. It defines an extensible computation graph model, as well as definitions of built-in operators and standard
data types. Currently we focus on the capabilities needed for inferencing (scoring).

ONNX is [widely supported](http://onnx.ai/supported-tools) and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community. We invite the community to join us and further evolve ONNX.

# Use ONNX
* [Tutorials for creating ONNX models](https://github.com/onnx/tutorials).
* [Pre-trained ONNX models](https://github.com/onnx/models)

# Learn about the ONNX spec
* [Overview](docs/Overview.md)
* [ONNX intermediate representation spec](docs/IR.md)
* [Versioning principles of the spec](docs/Versioning.md)
* [Operators documentation](docs/Operators.md)
* [Python API Overview](docs/PythonAPIOverview.md)

# Programming utilities for working with ONNX Graphs
* [Shape and Type Inference](docs/ShapeInference.md)
* [Graph Optimization](docs/Optimizer.md)
* [Opset Version Conversion](docs/VersionConverter.md)

# Contribute
ONNX is a [community project](community). We encourage you to join the effort and contribute feedback, ideas, and code. You can participate in the [SIGs](community/sigs.md) and [Working Groups](community/working-groups.md) to shape the future of ONNX.

Check out our [contribution guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) to get started.

If you think some operator should be added to ONNX specification, please read
[this document](docs/AddNewOp.md).

# Discuss
We encourage you to open [Issues](https://github.com/onnx/onnx/issues), or use Gitter for more real-time discussion:
[![Join the chat at https://gitter.im/onnx/Lobby](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/onnx/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# Follow Us
Stay up to date with the latest ONNX news. [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter](https://twitter.com/onnxai)]






# Installation

## Binaries

A binary build of ONNX is available from [Conda](https://conda.io), in [conda-forge](https://conda-forge.org/):

```
conda install -c conda-forge onnx
```

## Source

You will need an install of protobuf and numpy to build ONNX.  One easy
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

You can also build and install ONNX locally from source code:

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

After installation, run

```
python -c "import onnx"
```

to verify it works.  Note that this command does not work from
a source checkout directory; in this case you'll see:

```
ModuleNotFoundError: No module named 'onnx.onnx_cpp2py_export'
```

Change into another directory to fix this error.

# Testing

ONNX uses [pytest](https://docs.pytest.org) as test driver. In order to run tests, first you need to install pytest:

```
pip install pytest nbval
```

After installing pytest, do

```
pytest
```

to run tests.

# Development

Check out [contributor guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) for instructions.

# License

[MIT License](LICENSE)

# Code of Conduct

[ONNX Open Source Code of Conduct](https://onnx.ai/codeofconduct.html)
