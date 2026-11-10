```toml
name = "LRN"
domain = "ai.onnx"

[attributes.alpha]
type = "FLOAT"
desc = "Scaling parameter."
default = 0.0001

[attributes.beta]
type = "FLOAT"
desc = "The exponent."
default = 0.75

[attributes.bias]
type = "FLOAT"
desc = ""
default = 1.0

[attributes.size]
type = "INT"
desc = "The number of channels to sum over"
required = true

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]."

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor, which has the shape and type as input tensor"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output  types to float tensors."
```

Local Response Normalization proposed in the [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
It normalizes over local input regions.
The local region is defined across the channels. For an element X[n, c, d1, ..., dk] in a tensor
of shape (N x C x D1 x D2, ..., Dk), its region is
{X[n, i, d1, ..., dk] | max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2))}.

square_sum[n, c, d1, ..., dk] = sum(X[n, i, d1, ..., dk] ^ 2),
where max(0, c - floor((size - 1) / 2)) <= i <= min(C - 1, c + ceil((size - 1) / 2)).

Y[n, c, d1, ..., dk] = X[n, c, d1, ..., dk] / (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta
