
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
