```toml
name = "Split"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SplitShapeInference_opset13"

[attributes.axis]
type = "INT"
desc = "Which axis to split on. A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] where r = rank(input)."
default = 0

[[inputs]]
name = "input"
type = "T"
desc = "The tensor to split"
differentiation = "differentiable"

[[inputs]]
name = "split"
type = "tensor(int64)"
desc = "Optional length of each output. Values should be >= 0.Sum of the values must be equal to the dim value at 'axis' specified."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "outputs"
type = "T"
desc = "One or more outputs forming list of tensors after splitting"
option = "variadic"
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."
```

Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using input 'split'.
Otherwise, the tensor is split to equal sized parts.
