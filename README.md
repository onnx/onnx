Open Neural Network Exchange (ONNX)
========
[![Build Status](https://travis-ci.org/onnx/onnx.svg?branch=master)](https://travis-ci.org/onnx/onnx)
[![Build status](https://ci.appveyor.com/api/projects/status/0yglo4dgwpc1so3v/branch/master?svg=true)](https://ci.appveyor.com/project/marcelolr/onnx/branch/master)

Open Neural Network Exchange (ONNX) is the first step toward an open ecosystem that empowers AI developers
to choose the right tools as their project evolves. ONNX provides an open source format for AI models. 
It defines an extensible computation graph model, as well as definitions of built-in operators and standard 
data types. Initially we focus on the capabilities needed for inferencing (evaluation).

Caffe2, PyTorch, and Cognitive Toolkit will be supporting ONNX. Enabling interoperability between different 
frameworks and streamlining the path from research to production will increase the speed of innovation in 
the AI community. We are an early stage and we invite the community to submit feedback and help us further 
evolve ONNX.


# Folder Structure

- onnx/: the main folder that all code lies under
  - onnx.proto: the protobuf (v2.6.1) that contains all the structures
  - checker.py: utility to check whether a serialized ONNX proto is legal.
  - defs/: subfolder that defines the ONNX operators.
  - test/: test files

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

You can then install ONNX from PyPi:

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

You will need an install of protobuf and numpy to build ONNX.  One easy
way to get these dependencies is via
[Anaconda](https://www.anaconda.com/download/):

```
# Use conda-forge protobuf, as defaults doesn't come with protoc
conda install -c conda-forge protobuf numpy
```

During development it's convenient to install ONNX in development mode:

```
git clone --recursive https://github.com/onnx/onnx.git
pip install -e onnx/
```
Then, after you have made changes to

- Python files, the changes are immediately effective in your installation, you do not need to install again.
- C++ files, you need to do install again to trigger the native extension build.

## Generated operator documentation

[Operator docs in Operators.md](docs/Operators.md) are auto-generated based on C++ operator definitions. In order to refresh them run the following command from the repo root and commit the results:

```
python onnx/defs/gen_doc.py
```



# License

[MIT License](LICENSE)
