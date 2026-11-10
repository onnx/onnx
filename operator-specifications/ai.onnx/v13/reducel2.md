```toml
name = "ReduceL2"
domain = "ai.onnx"
shape_inference = "defs::reduction::utils::ReduceShapeInference"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."
differentiation = "differentiable"

[[outputs]]
name = "reduced"
type = "T"
desc = "Reduced output tensor."
differentiation = "differentiable"

[type_constraints.T]
types = "numeric_types_for_math_reduction_ir4"
desc = "Constrain input and output types to numeric tensors."
```

Computes the L2 norm of the input tensor's elements along the provided axes. The resulting
tensor has the same rank as the input if `keepdims` equals 1. If `keepdims` equals 0, then
the resulting tensor has the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields 0.


The above behavior is similar to numpy, with the exception that numpy defaults `keepdims`
to `False` instead of `True`.
