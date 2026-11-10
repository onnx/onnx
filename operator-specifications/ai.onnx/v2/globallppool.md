```toml
name = "GlobalLpPool"
domain = "ai.onnx"

[attributes.p]
type = "INT"
desc = "p value of the Lp norm used to pool over the input data."
default = 2

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size."
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output data tensor from pooling across the input tensor. The output tensor has the same rank as the input. The first two dimensions of output shape are the same as the input (N x C), while the other dimensions are all 1."
differentiation = "differentiable"
```

GlobalLpPool consumes an input tensor X and applies lp pool pooling across
the values in the same channel. This is equivalent to LpPool with kernel size
equal to the spatial dimension of input tensor.
