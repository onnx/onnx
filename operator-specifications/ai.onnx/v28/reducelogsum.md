```toml
name = "ReduceLogSum"
domain = "ai.onnx"
shape_inference = "defs::reduction::utils::ReduceShapeInference"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[function_bodies]
28 = """
{
   reduced_sum = ReduceSum <keepdims: int = @keepdims, noop_with_empty_axes: int = @noop_with_empty_axes> (data, axes)
   reduced = Log (reduced_sum)
}
"""
```
