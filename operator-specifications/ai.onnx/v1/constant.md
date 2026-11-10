```toml
name = "Constant"
domain = "ai.onnx"

[attributes.value]
type = "TENSOR"
desc = "The value for the elements of the output tensor."
required = true

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor containing the same value of the provided tensor."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

A constant tensor.
