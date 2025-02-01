# onnx.numpy_helper

```{eval-rst}
.. currentmodule:: onnx.numpy_helper
```

```{eval-rst}
.. autosummary::

    from_array
    from_dict
    from_list
    from_optional
    to_array
    to_dict
    to_list
    to_optional

```

(l-numpy-helper-onnx-array)=

## array

```{eval-rst}
.. autofunction:: onnx.numpy_helper.from_array
```

```{eval-rst}
.. autofunction:: onnx.numpy_helper.to_array
```

As numpy does not support all the types defined in ONNX (float 8 types, blofat16, int4, uint4, float4e2m1),
these two functions use a custom dtype defined in :mod:`onnx._custom_element_types`.

## sequence

```{eval-rst}
.. autofunction:: onnx.numpy_helper.to_list
```

```{eval-rst}
.. autofunction:: onnx.numpy_helper.from_list
```

## dictionary

```{eval-rst}
.. autofunction:: onnx.numpy_helper.to_dict
```

```{eval-rst}
.. autofunction:: onnx.numpy_helper.from_dict
```

## optional

```{eval-rst}
.. autofunction:: onnx.numpy_helper.to_optional
```

```{eval-rst}
.. autofunction:: onnx.numpy_helper.from_optional
```
