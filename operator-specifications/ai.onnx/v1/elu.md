```toml
name = "Elu"
domain = "ai.onnx"
shape_inference = false

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of ELU default to 1.0."
default = 1.0

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

Elu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x <
0`, `f(x) = x for x >= 0`., is applied to the tensor elementwise.
