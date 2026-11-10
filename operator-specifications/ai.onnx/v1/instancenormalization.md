```toml
name = "InstanceNormalization"
domain = "ai.onnx"
shape_inference = false

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero, default is 1e-5f."
default = 1e-05

[[inputs]]
name = "input"
type = "T"
desc = "The input 4-dimensional tensor of shape NCHW."

[[inputs]]
name = "scale"
type = "T"
desc = "The input 1-dimensional scale tensor of size C."

[[inputs]]
name = "B"
type = "T"
desc = "The input 1-dimensional bias tensor of size C."

[[outputs]]
name = "output"
type = "T"
desc = "The output 4-dimensional tensor of the same shape as input."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Carries out instance normalization as described in the paper
https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B,
where mean and variance are computed per instance per channel.
