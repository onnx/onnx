(l-mod-onnx-helper)=

# onnx.helper

```{eval-rst}
.. currentmodule:: onnx.helper
```

(l-onnx-make-function)=

## Helper functions to make ONNX graph components

All functions used to create an ONNX graph.

```{eval-rst}
.. autofunction:: onnx.helper.make_attribute
```

```{eval-rst}
.. autofunction:: onnx.helper.make_attribute_ref
```

```{eval-rst}
.. autofunction:: onnx.helper.make_empty_tensor_value_info
```

```{eval-rst}
.. autofunction:: onnx.helper.make_function
```

```{eval-rst}
.. autofunction:: onnx.helper.make_graph
```

```{eval-rst}
.. autofunction:: onnx.helper.make_map
```

```{eval-rst}
.. autofunction:: onnx.helper.make_map_type_proto
```

```{eval-rst}
.. autofunction:: onnx.helper.make_model
```

```{eval-rst}
.. autofunction:: onnx.helper.make_node
```

```{eval-rst}
.. autofunction:: onnx.helper.make_operatorsetid
```

```{eval-rst}
.. autofunction:: onnx.helper.make_opsetid
```

```{eval-rst}
.. autofunction:: onnx.helper.make_model_gen_version
```

```{eval-rst}
.. autofunction:: onnx.helper.make_optional
```

```{eval-rst}
.. autofunction:: onnx.helper.make_optional_type_proto
```

```{eval-rst}
.. autofunction:: onnx.helper.make_sequence
```

```{eval-rst}
.. autofunction:: onnx.helper.make_sequence_type_proto
```

```{eval-rst}
.. autofunction:: onnx.helper.make_sparse_tensor
```

```{eval-rst}
.. autofunction:: onnx.helper.make_sparse_tensor_type_proto
```

```{eval-rst}
.. autofunction:: onnx.helper.make_sparse_tensor_value_info
```

```{eval-rst}
.. autofunction:: onnx.helper.make_tensor
```

```{eval-rst}
.. autofunction:: onnx.helper.make_tensor_sequence_value_info
```

```{eval-rst}
.. autofunction:: onnx.helper.make_tensor_type_proto
```

```{eval-rst}
.. autofunction:: onnx.helper.make_training_info
```

```{eval-rst}
.. autofunction:: onnx.helper.make_tensor_value_info
```

```{eval-rst}
.. autofunction:: onnx.helper.make_value_info
```

## Type Mappings

```{eval-rst}
.. autofunction:: onnx.helper.get_all_tensor_dtypes
```

```{eval-rst}
.. autofunction:: onnx.helper.np_dtype_to_tensor_dtype
```

```{eval-rst}
.. autofunction:: onnx.helper.tensor_dtype_to_field
```

```{eval-rst}
.. autofunction:: onnx.helper.tensor_dtype_to_np_dtype
```

```{eval-rst}
.. autofunction:: onnx.helper.tensor_dtype_to_storage_tensor_dtype
```

```{eval-rst}
.. autofunction:: onnx.helper.tensor_dtype_to_string
```

## Tools

```{eval-rst}
.. autofunction:: onnx.helper.find_min_ir_version_for
```

```{eval-rst}
.. autofunction:: onnx.helper.create_op_set_id_version_map
```

## Other functions

```{eval-rst}
.. autosummary::

    get_attribute_value
    get_node_attr_value
    set_metadata_props
    set_model_props
    printable_attribute
    printable_dim
    printable_graph
    printable_node
    printable_tensor_proto
    printable_type
    printable_value_info
```
