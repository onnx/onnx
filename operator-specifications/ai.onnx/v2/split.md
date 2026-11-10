```toml
name = "Split"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SplitShapeInference_opset2"

[attributes.axis]
type = "INT"
desc = "Which axis to split on. "
default = 0

[attributes.split]
type = "INTS"
desc = "length of each output"

[[inputs]]
name = "input"
type = "T"
desc = "The tensor to split"

[[outputs]]
name = "outputs"
type = "T"
desc = "One or more outputs forming list of tensors after splitting"
option = "variadic"

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
