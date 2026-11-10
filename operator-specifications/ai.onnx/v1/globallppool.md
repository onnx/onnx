```toml
name = "GlobalLpPool"
domain = "ai.onnx"
shape_inference = false

[attributes.p]
type = "FLOAT"
desc = "p value of the Lp norm used to pool over the input data, default is 2.0."
default = 2.0

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimension are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size."

[[outputs]]
name = "Y"
type = "T"
desc = "Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

GlobalLpPool consumes an input tensor X and applies lp pool pooling across the
the values in the same channel. This is equivalent to LpPool with kernel size
equal to the spatial dimension of input tensor.
