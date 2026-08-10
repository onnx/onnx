(l-python-onnx-api)=

# API Reference

```{tip}
The [ir-py project](https://github.com/onnx/ir-py) provides alternative Pythonic APIs for creating and manipulating ONNX models without interaction with Protobuf.
```

## Versioning

The following example shows how to retrieve the installed ONNX package version,
the default ONNX opset version, and the IR version (see
{ref}`l-api-opset-version`).

```{eval-rst}
.. exec_code::

    from onnx import __version__, IR_VERSION
    from onnx.defs import onnx_opset_version
    print(f"onnx.__version__={__version__!r}, opset={onnx_opset_version()}, IR_VERSION={IR_VERSION}")
```

The intermediate representation (IR) specification defines the abstract model
for graphs and operators and the concrete format that represents them. The IR
version increases when that representation changes.

An opset version identifies a published set of operator schemas in a domain. It
increases when an operator is added, removed, or modified, for example to support
additional input or output types or to replace an attribute with an input.

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

An ONNX model can be created directly from the classes described
in the previous section, but it is faster to create and
verify a model with the following helpers.

```{toctree}
:maxdepth: 1

backend
checker
compose
defs
external_data_helper
helper
inliner
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
