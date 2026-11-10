```toml
name = "InstanceNormalization"
domain = "ai.onnx"

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero."
default = 1e-05

[[inputs]]
name = "input"
type = "T"
desc = "Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size."
differentiation = "differentiable"

[[inputs]]
name = "scale"
type = "T"
desc = "The input 1-dimensional scale tensor of size C."
differentiation = "differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "The input 1-dimensional bias tensor of size C."
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "The output tensor of the same shape as input."
differentiation = "differentiable"
```
