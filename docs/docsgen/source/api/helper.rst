
.. _l-mod-onnx-helper:

onnx.helper
===========


.. currentmodule:: onnx.helper

.. autosummary::

    find_min_ir_version_for
    get_all_tensor_dtypes
    get_attribute_value
    float32_to_bfloat16
    float32_to_float8e4m3
    float32_to_float8e5m2
    make_attribute
    make_empty_tensor_value_info
    make_function
    make_graph
    make_map
    make_model
    make_node
    make_operatorsetid
    make_opsetid
    make_optional
    make_optional_type_proto
    make_sequence
    make_sequence_type_proto
    make_sparse_tensor
    make_sparse_tensor_type_proto
    make_sparse_tensor_value_info
    make_tensor
    make_tensor_sequence_value_info
    make_tensor_type_proto
    make_training_info
    make_tensor_type_proto
    make_tensor_value_info
    make_value_info
    np_dtype_to_tensor_dtype
    printable_attribute
    printable_dim
    printable_graph
    printable_node
    printable_tensor_proto
    printable_type
    printable_value_info
    split_complex_to_pairs
    tensor_dtype_to_np_dtype
    tensor_dtype_to_storage_tensor_dtype
    tensor_dtype_to_string
    tensor_dtype_to_field

getter
++++++

.. autofunction:: onnx.helper.get_attribute_value

print
+++++

.. autofunction:: onnx.helper.printable_attribute

.. autofunction:: onnx.helper.printable_dim

.. autofunction:: onnx.helper.printable_graph

.. autofunction:: onnx.helper.printable_node

.. autofunction:: onnx.helper.printable_tensor_proto

.. autofunction:: onnx.helper.printable_type

.. autofunction:: onnx.helper.printable_value_info

tools
+++++

.. autofunction:: onnx.helper.find_min_ir_version_for

.. autofunction:: onnx.helper.split_complex_to_pairs

.. _l-onnx-make-function:

make function
+++++++++++++

All functions uses to create an ONNX graph.

.. autofunction:: onnx.helper.make_attribute

.. autofunction:: onnx.helper.make_empty_tensor_value_info

.. autofunction:: onnx.helper.make_function

.. autofunction:: onnx.helper.make_graph

.. autofunction:: onnx.helper.make_map

.. autofunction:: onnx.helper.make_model

.. autofunction:: onnx.helper.make_node

.. autofunction:: onnx.helper.make_operatorsetid

.. autofunction:: onnx.helper.make_opsetid

.. autofunction:: onnx.helper.make_optional

.. autofunction:: onnx.helper.make_optional_type_proto

.. autofunction:: onnx.helper.make_sequence

.. autofunction:: onnx.helper.make_sequence_type_proto

.. autofunction:: onnx.helper.make_sparse_tensor

.. autofunction:: onnx.helper.make_sparse_tensor_type_proto

.. autofunction:: onnx.helper.make_sparse_tensor_value_info

.. autofunction:: onnx.helper.make_tensor

.. autofunction:: onnx.helper.make_tensor_sequence_value_info

.. autofunction:: onnx.helper.make_tensor_type_proto

.. autofunction:: onnx.helper.make_training_info

.. autofunction:: onnx.helper.make_tensor_type_proto

.. autofunction:: onnx.helper.make_tensor_value_info

.. autofunction:: onnx.helper.make_value_info

getter
++++++

.. autofunction:: onnx.helper.get_attribute_value

print
+++++

.. autofunction:: onnx.helper.printable_attribute

.. autofunction:: onnx.helper.printable_dim

.. autofunction:: onnx.helper.printable_graph

.. autofunction:: onnx.helper.printable_node

.. autofunction:: onnx.helper.printable_tensor_proto

.. autofunction:: onnx.helper.printable_type

.. autofunction:: onnx.helper.printable_value_info

type mappings
+++++++++++++

.. autofunction:: onnx.helper.get_all_tensor_dtypes

.. autofunction:: onnx.helper.np_dtype_to_tensor_dtype

.. autofunction:: onnx.helper.tensor_dtype_to_field

.. autofunction:: onnx.helper.tensor_dtype_to_np_dtype

.. autofunction:: onnx.helper.tensor_dtype_to_storage_tensor_dtype

.. autofunction:: onnx.helper.tensor_dtype_to_string

cast
++++

.. autofunction:: onnx.helper.float32_to_bfloat16

.. autofunction:: onnx.helper.float32_to_float8e4m3

.. autofunction:: onnx.helper.float32_to_float8e5m2
