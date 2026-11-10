```toml
name = "Selu"
domain = "ai.onnx"
shape_inference = false

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of SELU default to 1.6732."
default = 1.6732

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[attributes.gamma]
type = "FLOAT"
desc = "Coefficient of SELU default to 1.0507."
default = 1.0507

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

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.
