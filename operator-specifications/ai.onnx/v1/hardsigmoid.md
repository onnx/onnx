```toml
name = "HardSigmoid"
domain = "ai.onnx"

[attributes.alpha]
type = "FLOAT"
desc = "Value of alpha default to 0.2"
default = 0.2

[attributes.beta]
type = "FLOAT"
desc = "Value of beta default to 0.5"
default = 0.5

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

HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.
