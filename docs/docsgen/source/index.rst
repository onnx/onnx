.. api-content documentation master file, created by
   sphinx-quickstart on Mon Jun 20 10:52:22 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ONNX static content generation
==============================

This is a developer usage guide to the ONNX Python API and Operator Schemas.
It contains the following information for the latest release:

.. exec_code::

    import onnx
    from onnx.defs import onnx_opset_version
    print(f"onnx.__version__: {onnx.__version__!r}")
    print(f"onnx opset: {onnx_opset_version()}")

.. toctree::
    :maxdepth: 2

    onnx_doc_folder/index
    onnx_python/index
