

<p align="center"><img width="40%" src="docs/ONNX_logo_main.png" /></p>

[![Build Status](https://travis-ci.org/onnx/onnx.svg?branch=master)](https://travis-ci.org/onnx/onnx)
[![Build status](https://ci.appveyor.com/api/projects/status/lm50cevk2hmrll98?svg=true)](https://ci.appveyor.com/project/onnx/onnx)

[Open Neural Network Exchange (ONNX)](http://onnx.ai) is the first step toward an open ecosystem that empowers AI developers
to choose the right tools as their project evolves. ONNX provides an open source format for AI models. 
It defines an extensible computation graph model, as well as definitions of built-in operators and standard 
data types. Initially we focus on the capabilities needed for inferencing (evaluation).

Caffe2, PyTorch, Cognitive Toolkit will, Apache MXNet and other tools are developing ONNX support. Enabling interoperability between different 
frameworks and streamlining the path from research to production will increase the speed of innovation in 
the AI community. We are an early stage and we invite the community to submit feedback and help us further 
evolve ONNX.

# Use ONNX
Start experimenting today:
* [Getting Started Guide](http://onnx.ai/getting-started)
* [Supported Frameworks & Tools](http://onnx.ai/supported-tools)
* [Tutorials on using ONNX converters](https://github.com/onnx/tutorials).

# Learn about ONNX spec

Check ONNX design choices and internals:
* [Overview](docs/Overview.md)
* [ONNX intermediate representation spec](docs/IR.md)
* [Versioning principles of the spec](docs/Versioning.md)
* [Operators documentation](docs/Operators.md)

# Contribute
ONNX is a community project.  We encourage you to join the effort and contribute feedback, ideas, and code. Check out our [contribution guide](https://github.com/onnx/onnx/blob/master/docs/CONTRIBUTING.md) to get started.

# Follow Us
Stay up to date with the latest ONNX news. [[Facebook](https://www.facebook.com/onnxai/)] [[Twitter](https://twitter.com/onnxai)]






# Installation

## Binaries

A binary build of ONNX is available from [Conda](https://conda.io):

```
conda install -c ezyang onnx
```

## Docker

Docker images (CPU-only and GPU versions) with ONNX, PyTorch, and Caffe2 are availiable for quickly trying [tutorials that use ONNX](http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html). To quickly try CPU-only version, simply run:

```
docker run -it --rm onnx/onnx-docker:cpu /bin/bash
```

To run the version with GPU support, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is needed. Execute:
```
nvidia-docker run -it --rm onnx/onnx-docker:gpu /bin/bash
```

## Source

You will need an install of protobuf and numpy to build ONNX.  One easy
way to get these dependencies is via
[Anaconda](https://www.anaconda.com/download/):

```
# Use conda-forge protobuf, as defaults doesn't come with protoc
conda install -c conda-forge protobuf numpy
```

You can then install ONNX from PyPi (Note: Add install option --install-option="--onnxml=1" for onnx-ml):

```
pip install onnx
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
pip install pytest-cov nbval
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

[ONNX Open Source Code of Conduct](http://onnx.ai/codeofconduct.html)
