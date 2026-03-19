(l-serialization)=

# Serialization

## Save a model and any Proto class

This ONNX graph needs to be serialized into one contiguous
memory buffer. Method `SerializeToString` is available
in every ONNX objects.

```
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

This method has the following signature.

```{eval-rst}
.. autoclass:: onnx.ModelProto
    :members: SerializeToString
```

Every Proto class implements method `SerializeToString`.
Therefore the following code works with any class described
in page {ref}`l-onnx-classes`.

```
with open("proto.pb", "wb") as f:
    f.write(proto.SerializeToString())
```

Next example shows how to save a {ref}`l-nodeproto`.

```{eval-rst}
.. exec_code::

    from onnx import NodeProto

    node = NodeProto()
    node.name = "example-type-proto"
    node.op_type = "Add"
    node.input.extend(["X", "Y"])
    node.output.extend(["Z"])

    with open("node.pb", "wb") as f:
        f.write(node.SerializeToString())
```

## Load a model

Following function only automates the loading of a class
{ref}`l-modelproto`. Next sections shows how to restore
any other proto class.

```{eval-rst}
.. autofunction:: onnx.load
```

```
from onnx import load

onnx_model = load("model.onnx")
```

Or:

```
from onnx import load

with open("model.onnx", "rb") as f:
    onnx_model = load(f)
```

Next function does the same from a bytes array.

```{eval-rst}
.. autofunction:: onnx.load_model_from_string

```

(l-onnx-load-data)=

## Load a Proto

Proto means here any type containing data including a model, a tensor,
a sparse tensor, any class listed in page {ref}`l-onnx-classes`.
The user must know the type of the data he needs to restore
and then call method `ParseFromString`.
[protobuf](https://developers.google.com/protocol-buffers)
does not store any information about the class
of the saved data. Therefore, this class must be known before
restoring an object.

```{eval-rst}
.. autoclass:: onnx.ModelProto
    :members: ParseFromString
```

Next example shows how to restore a {ref}`l-nodeproto`.

```{eval-rst}
.. exec_code::

    from onnx import NodeProto

    tp2 = NodeProto()
    with open("node.pb", "rb") as f:
        content = f.read()

    tp2.ParseFromString(content)

    print(tp2)
```

A shortcut exists for {ref}`l-tensorproto`:

```{eval-rst}
.. autofunction:: onnx.load_tensor_from_string
```
