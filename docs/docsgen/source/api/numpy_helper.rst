
onnx.numpy_helper
=================


.. currentmodule:: onnx.numpy_helper

.. autosummary::

    bfloat16_to_float32
    float8e4m3_to_float32
    float8e5m2_to_float32
    from_array
    from_dict
    from_list
    from_optional
    to_array
    to_dict
    to_list
    to_optional


.. _l-numpy-helper-onnx-array:

array
+++++

.. autofunction:: onnx.numpy_helper.from_array

.. autofunction:: onnx.numpy_helper.to_array

sequence
++++++++

.. autofunction:: onnx.numpy_helper.to_list

.. autofunction:: onnx.numpy_helper.from_list

dictionary
++++++++++

.. autofunction:: onnx.numpy_helper.to_dict

.. autofunction:: onnx.numpy_helper.from_dict

optional
++++++++

.. autofunction:: onnx.numpy_helper.to_optional

.. autofunction:: onnx.numpy_helper.from_optional

tools
+++++

.. autofunction:: onnx.numpy_helper.convert_endian

.. autofunction:: onnx.numpy_helper.combine_pairs_to_complex
    
.. autofunction:: onnx.numpy_helper.create_random_int

cast
++++

.. autofunction:: onnx.numpy_helper.bfloat16_to_float32

.. autofunction:: onnx.numpy_helper.float8e4m3_to_float32

.. autofunction:: onnx.numpy_helper.float8e5m2_to_float32
