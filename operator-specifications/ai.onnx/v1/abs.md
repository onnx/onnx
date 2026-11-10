```toml
name = "Abs"
domain = "ai.onnx"

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Absolute takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the absolute is, y = abs(x), is applied to
the tensor elementwise.
