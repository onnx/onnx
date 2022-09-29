
.. _l-mod-onnx-mapping:

onnx.mapping
============

This module defines the correspondance between onnx numerical types
and numpy numerical types. This information can be accessed
through attribute :ref:`l-onnx-types-mapping` or through the functions 
defined in :ref:`l-mod-onnx-helper`.

.. contents::
    :local:

TensorDtypeMap
++++++++++++++

.. autoclass:: onnx.mapping.TensorDtypeMap

.. _l-onnx-types-mapping:

TENSOR_TYPE_MAP
+++++++++++++++

.. exec_code::

    import pprint
    from onnx.mapping import TENSOR_TYPE_MAP

    pprint.pprint(TENSOR_TYPE_MAP)

Opset Version
+++++++++++++

.. autofunction:: onnx.defs.onnx_opset_version

.. autofunction:: onnx.defs.get_all_schemas_with_history

Operators and Functions Schemas
+++++++++++++++++++++++++++++++

.. autofunction:: onnx.defs.get_function_ops

.. autofunction:: onnx.defs.get_schema

Internal module
+++++++++++++++

.. automodule:: onnx.onnx_cpp2py_export.defs
    :members:
