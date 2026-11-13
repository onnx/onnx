# onnx.checker

## Checking models with external data

Check a model with external data from its path. External data locations are relative to the model directory, so a model checked in memory fails if it still references external data, for example one loaded with `load_external_data=False`. A model larger than 2GB must be checked from its path in any case.

```python
import onnx

onnx.checker.check_model("path/to/the/model.onnx")
```

By default, the checker verifies that each external data file exists, is a regular file inside the model directory, is not a symbolic link or hard link, and holds enough data for its tensors.

### Checking a model without its external data

With `check_external_data_files=False`, the checker does not access the external data files and checks only what the model itself says about them: every externally stored tensor has a `location` that is a relative path naming a file inside the model directory, `offset` and `length` are non-negative integers whose sum fits a file, and an explicit `length` is large enough for the tensor's shape and type. This checks a model whose external data has not been downloaded:

```python
import onnx

onnx.checker.check_model("path/to/the/model.onnx", check_external_data_files=False)
```

On the command line, `check-model --no-check-external-data-files path/to/the/model.onnx` does the same.

Without the files, the containment, symbolic link and hard link checks described in [External Data Security](../../../ExternalDataSecurity.md) cannot run, so a model that passes this check can still be rejected once its external data is present.

The checker never reads external data, so it cannot check the order and range of sparse tensor indices that are stored externally. With `check_external_data_files=False`, only the type, rank and dimensions of such indices are checked; by default, they are rejected. Indices stored inline are checked as usual.

## CheckerContext

```{eval-rst}
.. autoclass:: onnx.checker.DEFAULT_CONTEXT
    :members:
```

## The `onnx.checker` module

```{eval-rst}
.. automodule:: onnx.checker
    :members:
    :undoc-members:
    :show-inheritance:
```
