
.. _l-onnx-classes:

=======================
Proto and Serialization
=======================

.. contents::
    :local:

Proto
=====

AttributeProto
++++++++++++++

.. autoclass:: onnx.AttributeProto
    :members:

.. _l-onnx-function-proto:

FunctionProto
+++++++++++++

.. autoclass:: onnx.FunctionProto
    :members:

.. _l-onnx-graph-proto:

GraphProto
++++++++++

.. autoclass:: onnx.GraphProto
    :members:

.. _l-onnx-map-proto:

MapProto
++++++++

.. autoclass:: onnx.MapProto
    :members:

.. _l-modelproto:

ModelProto
++++++++++

.. autoclass:: onnx.ModelProto
    :members:

.. _l-nodeproto:

NodeProto
+++++++++

.. autoclass:: onnx.NodeProto
    :members:

.. _l-operatorproto:

OperatorProto
+++++++++++++

.. autoclass:: onnx.OperatorProto
    :members:

.. _l-operatorsetidproto:

OperatorSetIdProto
++++++++++++++++++

.. autoclass:: onnx.OperatorSetIdProto
    :members:

.. _l-operatorsetproto:

OperatorSetProto
++++++++++++++++

.. autoclass:: onnx.OperatorSetProto
    :members:

.. _l-optionalproto:

OptionalProto
+++++++++++++

.. autoclass:: onnx.OptionalProto
    :members:

.. _l-onnx-sequence-proto:

SequenceProto
+++++++++++++

.. autoclass:: onnx.SequenceProto
    :members:

.. _l-onnx-sparsetensor-proto:

SparseTensorProto
+++++++++++++++++

.. autoclass:: onnx.SparseTensorProto
    :members:

StringStringEntryProto

.. _l-tensorproto:

TensorProto
+++++++++++

.. autoclass:: onnx.TensorProto
    :members:

.. _l-tensorshapeproto:

TensorShapeProto
++++++++++++++++

.. autoclass:: onnx.TensorShapeProto
    :members:

.. _l-traininginfoproto:

TrainingInfoProto
+++++++++++++++++

.. autoclass:: onnx.TrainingInfoProto
    :members:

.. _l-typeproto:

TypeProto
+++++++++

.. autoclass:: onnx.TypeProto
    :members:

.. _l-valueinfoproto:

ValueInfoProto
++++++++++++++

.. autoclass:: onnx.ValueInfoProto
    :members:

Serialization
=============

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

.. _l-onnx-load-data:

Load data
+++++++++

Data means here any type containing data including a model, a tensor,
a sparse tensor...

.. autofunction:: onnx.load_model_from_string

.. autofunction:: onnx.load_tensor_from_string

`protobuf <https://developers.google.com/protocol-buffers>`_
does not store any information about the class
of the saved data. Therefore, this class must be known before
restoring an object.

Save data
+++++++++

Any `Proto` class includes a method called `SerializeToString`.
It must be called to serialize any proto object into
an array of bytes.

.. autoclass:: onnx.TensorProto
    :members: SerializeToString
