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

Arrays with data types not supported natively by NumPy will be return with ``ml_dtypes`` dtypes.

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
