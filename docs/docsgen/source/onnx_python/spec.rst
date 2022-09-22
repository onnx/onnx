
Specifications
==============

.. contents::
    :local:

Type Mappings
+++++++++++++

.. _l-onnx-types-mapping:

NP_TYPE_TO_TENSOR_TYPE
~~~~~~~~~~~~~~~~~~~~~~

.. exec_code::

    import pprint
    from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

    pprint.pprint(NP_TYPE_TO_TENSOR_TYPE)

OP_SET_ID_VERSION_MAP
~~~~~~~~~~~~~~~~~~~~~

.. exec_code::

    import pprint
    from onnx.helper import OP_SET_ID_VERSION_MAP

    pprint.pprint(OP_SET_ID_VERSION_MAP)

OPTIONAL_ELEMENT_TYPE_TO_FIELD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. exec_code::

    import pprint
    from onnx.mapping import OPTIONAL_ELEMENT_TYPE_TO_FIELD

    pprint.pprint(OPTIONAL_ELEMENT_TYPE_TO_FIELD)

STORAGE_ELEMENT_TYPE_TO_FIELD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. exec_code::

    import pprint
    from onnx.mapping import STORAGE_ELEMENT_TYPE_TO_FIELD

    pprint.pprint(STORAGE_ELEMENT_TYPE_TO_FIELD)

STORAGE_TENSOR_TYPE_TO_FIELD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. exec_code::

    import pprint
    from onnx.mapping import STORAGE_TENSOR_TYPE_TO_FIELD

    pprint.pprint(STORAGE_TENSOR_TYPE_TO_FIELD)

TENSOR_TYPE_TO_NP_TYPE
~~~~~~~~~~~~~~~~~~~~~~

.. exec_code::

    import pprint
    from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

    pprint.pprint(TENSOR_TYPE_TO_NP_TYPE)

TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. exec_code::

    import pprint
    from onnx.mapping import TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE

    pprint.pprint(TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE)

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
