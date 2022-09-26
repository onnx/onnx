ONNX documentation
==================

`Open Neural Network Exchange (ONNX) <https://onnx.ai/>`_
is an open ecosystem that empowers
AI developers to choose the right tools as their project evolves.
ONNX provides an open source format for AI models, 
both deep learning and traditional ML.
It defines an extensible computation graph model,
as well as definitions of built-in operators and standard data types.

This documentation introduces the Python package
`onnx <https://github.com/onnx/onnx>`_. A tutorial shows how
to build an ONNX graph through the Python API. This graph can then
be consumed by any runtime implementing ONNX specifications
described in last section. It lists all existing operators
in following version and below.

.. exec_code::

    import onnx
    from onnx.defs import onnx_opset_version
    print(f"onnx.__version__: {onnx.__version__!r}")
    print(f"onnx opset: {onnx_opset_version()}")

.. toctree::
    :maxdepth: 2

    tutorial_python/index
    onnx_python/index
    onnx_doc_folder/index
