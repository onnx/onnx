Open Neural Network Exchange (ONNX)
========

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

A binary build of ONNX is available from Conda:

```
conda install -c ezyang onnx
```

## Source

You can build ONNX from source with PIP:

```
git clone --recursive https://github.com/onnx/onnx.git
pip install onnx/
```

# Testing

ONNX uses [pytest](https://docs.pytest.org) as test driver. In order to run tests, first you need to install pytest:

```
pip install pytest-cov
```

After installing pytest, do

```
pytest
```

to run tests.

# Development

During development it's convenient to install ONNX in development mode:

```
git clone --recursive https://github.com/onnx/onnx.git
pip install -e onnx/
```
Then, after you have made changes to

- Python files, the changes are immediatly effective in your installation, you do not need to install again.
- C++ files, you need to do install again to trigger the native extension build.

# License

[MIT License](LICENSE)
