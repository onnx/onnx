```toml
name = "Sum"
domain = "ai.onnx"

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[[inputs]]
name = "data_0"
type = "T"
desc = "List of tensors for Sum."
option = "variadic"

[[outputs]]
name = "sum"
type = "T"
desc = "Output tensor. Same dimension as inputs."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Element-wise sum of each of the input tensors. All inputs and outputs must
have the same shape and data type.
