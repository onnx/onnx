
Serialization
=============

.. contents::
    :local:

Load a model
++++++++++++

.. autofunction:: onnx.load

::

    from onnx import load

    onnx_model = load("model.onnx")

Or:

::

    from onnx import load

    with open("model.onnx", "rb") as f:
        onnx_model = load(f)

Save a model
++++++++++++

This ONNX graph needs to be serialized into one contiguous
memory buffer. Method `SerializeToString` is available
in every ONNX objects.

::

    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

This method has the following signature.

.. autoclass:: onnx.ModelProto
    :members: SerializeToString

Load data
+++++++++

Data means here any type containing data including a model, a tensor,
a sparse tensor...

.. autofunction:: onnx.load_model_from_string

.. autofunction:: onnx.load_tensor_from_string

:epkg:`protobuf` does not store any information about the class
of the saved data. Therefore, this class must be known before
restoring an object.

Save data
+++++++++

Any `Proto` class includes a method called `SerializeToString`.
It must be called to serialize any :epkg:`onnx` object into
an array of bytes.

.. autoclass:: onnx.TensorProto
    :members: SerializeToString

Performance
+++++++++++

.. toctree::

    ../../gyexamples/plot_benchmark_onnx_serialize
