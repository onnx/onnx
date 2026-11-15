(l-python-onnx-api)=

# API Reference

```{tip}
For creating and manipulating ONNX models with a higher-level, more ergonomic Python interface without direct Protobuf interaction, consider:
- **[ir-py](https://github.com/onnx/ir-py)**: Modern in-memory IR and Python API for ONNX without raw Protobuf interaction.
- **[spox](https://github.com/Quantco/spox)**: Statically typed, functional API for composing ONNX models.
- **[ndonnx](https://github.com/Quantco/ndonnx)**: NumPy-compatible array interface that traces operations into ONNX graphs.
```

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
Adding a structure or modifying one of them increases the IR version.

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
