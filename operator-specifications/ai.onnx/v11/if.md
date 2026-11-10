```toml
name = "If"
domain = "ai.onnx"
shape_inference = "defs::controlflow::utils::IfInferenceFunction_opset11"

[[outputs]]
name = "outputs"
type = "V"
desc = "Values that are live-out to the enclosing scope. The return values in the `then_branch` and `else_branch` must be of the same data type. The `then_branch` and `else_branch` may produce tensors with the same element type and different shapes. If corresponding outputs from the then-branch and the else-branch have static shapes S1 and S2, then the shape of the corresponding output variable of the if-node (if present) must be compatible with both S1 and S2 as it represents the union of both possible shapes.For example, if in a model file, the first output of `then_branch` is typed float tensor with shape [2] and the first output of `else_branch` is another float tensor with shape [3], If's first output should have (a) no shape set, or (b) a shape of rank 1 with neither `dim_value` nor `dim_param` set, or (c) a shape of rank 1 with a unique `dim_param`. In contrast, the first output cannot have the shape [2] since [2] and [3] are not compatible."
option = "variadic"
homogeneous = false
```
