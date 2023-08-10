(l-reference-implementation)=

# onnx.reference

## DefaultNone

```{eval-rst}
.. autoclass:: onnx.reference.op_run.DefaultNone
    :members:
```

## Inference

```{eval-rst}
.. autoclass:: onnx.reference.ReferenceEvaluator
    :members: input_names, output_names, opsets, run
```

## OpFunction

```{eval-rst}
.. autoclass:: onnx.reference.op_run.OpFunction
    :members: create, eval, input, output, implicit_inputs, domain, need_context, run, make_node
```

## OpRun

```{eval-rst}
.. autoclass:: onnx.reference.op_run.OpRun
    :members: create, eval, input, output, implicit_inputs, domain, need_context, run, make_node
```

## RuntimeTypeError

```{eval-rst}
.. autoclass:: onnx.reference.op_run.RuntimeTypeError
    :members:
```

## SparseTensor

```{eval-rst}
.. autoclass:: onnx.reference.op_run.SparseTensor
    :members:
```
