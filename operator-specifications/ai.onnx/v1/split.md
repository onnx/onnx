```toml
name = "Split"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "Which axis to split on"

[attributes.split]
type = "INTS"
desc = "length of each output"

[[inputs]]
name = "input"
type = "T"
desc = "The tensor to split"

[[inputs]]
name = "split"
type = "T"
desc = "Optional list of output lengths (see also arg 'split')"
option = "optional"

[[outputs]]
name = "outputs..."
type = "T"
desc = "One or more outputs forming list of tensors after splitting"
option = "variadic"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input types to float tensors."
```

Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
