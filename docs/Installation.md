# Installation

## Binaries

A binary build of ONNX is available from [Conda](https://conda.io):

```
conda install -c ezyang onnx
```

## Docker

Docker images (CPU-only and GPU versions) with ONNX, PyTorch, and Caffe2 are available to quickly try [tutorials that use ONNX](http://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html). For the CPU-only version, simply run:

```
docker run -it --rm onnx/onnx-docker:cpu /bin/bash
```

To run the version with GPU support, [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is needed. Run:
```
nvidia-docker run -it --rm onnx/onnx-docker:gpu /bin/bash
```

## Source

You will need to install Protobuf and NumPy to build ONNX. An easy
way to get these dependencies is via [Anaconda](https://www.anaconda.com/download/):

```
# Use conda-forge protobuf, as defaults doesn't come with protoc
conda install -c conda-forge protobuf numpy
```

You can then install ONNX from PyPi (Note: set environment variable `ONNX_ML=1` for onnx-ml):

```
pip install onnx
```

Note: When installing in a non-Anaconda environment, make sure to install the Protobuf compiler before running the pip installation of ONNX. For example, on Ubuntu:

```
sudo apt-get install protobuf-compiler libprotoc-dev
pip install onnx
```

After installation, run

```
python -c "import onnx"
```

to verify that it works.  Note that this command does not work from
a source checkout directory. If working from a source checkout directory, you'll see:

```
ModuleNotFoundError: No module named 'onnx.onnx_cpp2py_export'
```

Change into another directory to fix this error.
