
.. _l-python-onnx-api:

ONNX API
========

The following example shows how to retrieve onnx version
and onnx opset. Every new major release increments the opset version.

.. autofunction:: onnx.defs.onnx_opset_version

.. exec_code::

    from onnx import __version__
    from onnx.defs import onnx_opset_version
    print("onnx", __version__, "opset", onnx_opset_version())

.. toctree::
    :maxdepth: 1

    classes
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
    version
