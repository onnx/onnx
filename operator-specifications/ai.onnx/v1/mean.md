```toml
name = "Mean"
domain = "ai.onnx"

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[[inputs]]
name = "data_0"
type = "T"
desc = "List of tensors for Mean."
option = "variadic"

[[outputs]]
name = "mean"
type = "T"
desc = "Output tensor. Same dimension as inputs."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Element-wise mean of each of the input tensors. All inputs and outputs must
have the same shape and data type.
