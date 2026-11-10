```toml
name = "ReduceLogSumExp"
domain = "ai.onnx"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[function_bodies]
28 = """
{
   data_double = Cast <to: int = 11> (data)
   data_exp = Exp (data_double)
   reduced_sum = ReduceSum <keepdims: int = @keepdims, noop_with_empty_axes: int = @noop_with_empty_axes> (data_exp, axes)
   reduced_double = Log (reduced_sum)
   reduced = CastLike (reduced_double, data)
}
"""
```
