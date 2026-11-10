```toml
name = "Log"
domain = "ai.onnx"

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"

[[outputs]]
name = "output"
type = "T"
desc = "The natural log of the input tensor computed element-wise"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Calculates the natural log of the given input tensor, element-wise.
