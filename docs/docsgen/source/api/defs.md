(l-mod-onnx-defs)=

# onnx.defs

(l-api-opset-version)=

## Opset Version

```{eval-rst}
.. autofunction:: onnx.defs.onnx_opset_version
```

```{eval-rst}
.. autofunction:: onnx.defs.get_all_schemas_with_history
```

## Operators and Functions Schemas

```{eval-rst}
.. autofunction:: onnx.defs.get_function_ops
```

```{eval-rst}
.. autofunction:: onnx.defs.get_schema
```

## class OpSchema

```{eval-rst}
.. autoclass:: onnx.defs.OpSchema
    :members:
```

## Exception SchemaError

```{eval-rst}
.. autoclass:: onnx.defs.SchemaError
    :members:
```

## Constants

Domains officially supported in onnx package.

```{eval-rst}
.. exec_code::

    from onnx.defs import (
        ONNX_DOMAIN,
        ONNX_ML_DOMAIN,
        AI_ONNX_PREVIEW_TRAINING_DOMAIN,
    )
    print(f"ONNX_DOMAIN={ONNX_DOMAIN!r}")
    print(f"ONNX_ML_DOMAIN={ONNX_ML_DOMAIN!r}")
    print(f"AI_ONNX_PREVIEW_TRAINING_DOMAIN={AI_ONNX_PREVIEW_TRAINING_DOMAIN!r}")
```
