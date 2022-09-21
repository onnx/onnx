
.. _l-python-onnx-api:

Summary of onnx API
===================

This section gathers many functions or
classes from :epkg:`onnx` used when generated ONNX files
from machine learned models. Most of the examples
are executed during the generation of the documenation
with this version of :epkg:`onnx`.

.. autofunction:: onnx.defs.onnx_opset_version

.. runpython::
    :showcode:

    from onnx import __version__
    from onnx.defs import onnx_opset_version
    print("onnx", __version__, "opset", onnx_opset_version())

Other functions are dispatched accress following sections.

.. toctree::
    :maxdepth: 1

    serialize
    helper
    numpy_helper
    classes
    shape_inference
    potting
    spec
    hub
    utils
