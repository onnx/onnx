
.. _l-python-onnx-api:

API Reference
=============

The following example shows how to retrieve onnx version
and onnx opset. Every new major release increments the opset version.

.. autofunction:: onnx.defs.onnx_opset_version

.. exec_code::

    from onnx import __version__
    from onnx.defs import onnx_opset_version
    print("onnx", __version__, "opset", onnx_opset_version())

Data Structures
+++++++++++++++

Every ONNX object is defined based on a `protobuf message
<https://googleapis.dev/python/protobuf/latest/google/protobuf/message.html>`_
and has a name ended with suffix `Proto`. For example, :ref:`l-nodeproto` defines
an operator, :ref:`l-tensorproto` defines a tensor. Next page lists all of them.


.. toctree::
    :maxdepth: 1

    classes
    serialization

Functions
+++++++++

An ONNX model can be directly from the classes described
in previous section but it is faster to create and
verify a model with the following helpers.

.. toctree::
    :maxdepth: 1

    backend
    checker
    compose
    external_data_helper
    helper
    hub
    mapping
    numpy_helper
    parser
    printer
    shape_inference
    tools
    utils
    version_converter
