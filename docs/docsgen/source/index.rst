.. api-content documentation master file, created by
   sphinx-quickstart on Mon Jun 20 10:52:22 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ONNX documentation
==================

.. exec_code::

<<<<<<< HEAD
    import onnx
    from onnx.defs import onnx_opset_version
    print(f"onnx.__version__: {onnx.__version__!r}")
    print(f"onnx opset: {onnx_opset_version()}")
=======
.. toctree::
    :maxdepth: 1

    onnx-api/index
    onnx_python/index

**Operators and Op Schemas**

All examples end by calling function ``expect`` which
checks a runtime produces the expected output for this example.
One implementation can be found in the first page
linked below.
>>>>>>> cdfba48f5c6a8c5f17efebbab3b5c0f258c70202

.. toctree::
    :maxdepth: 2

    onnx_doc_folder/index
<<<<<<< HEAD
    tutorial_python/index
    onnx_python/index
=======

>>>>>>> cdfba48f5c6a8c5f17efebbab3b5c0f258c70202
