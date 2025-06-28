(l-python-onnx-api)=

# API Reference

## Versioning

The following example shows how to retrieve onnx version,
the onnx opset, the IR version. Every new major release increments the opset version
(see {ref}`l-api-opset-version`).

```{eval-rst}
.. exec_code::

    from onnx import __version__, IR_VERSION
    from onnx.defs import onnx_opset_version
    print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")
```

The intermediate representation (IR) specification is the abstract model for
graphs and operators and the concrete format that represents them.
Adding a structure, modifying one them increases the IR version.

The opset version increases when an operator is added or removed or modified.
A higher opset means a longer list of operators and more options to
implement an ONNX functions. An operator is usually modified because it
supports more input and output type, or an attribute becomes an input.

## Data Structures

Every ONNX object is defined based on a [protobuf message](https://googleapis.dev/python/protobuf/latest/google/protobuf/message.html)
and has a name ended with suffix `Proto`. For example, {ref}`l-nodeproto` defines
an operator, {ref}`l-tensorproto` defines a tensor. Next page lists all of them.

```{toctree}
:maxdepth: 1

classes
serialization
```

## Functions

An ONNX model can be directly from the classes described
in previous section but it is faster to create and
verify a model with the following helpers.

```{toctree}
:maxdepth: 1

backend
checker
compose
defs
external_data_helper
helper
hub
inliner
mapping
model_container
numpy_helper
parser
printer
reference
shape_inference
tools
utils
version_converter
```
