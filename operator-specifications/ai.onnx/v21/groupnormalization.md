```toml
name = "GroupNormalization"
domain = "ai.onnx"
deprecated = false

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero."
default = 1e-05

[attributes.num_groups]
type = "INT"
desc = "The number of groups of channels. It should be a divisor of the number of channels `C`."
required = true

[attributes.stash_type]
type = "INT"
desc = "The floating-point precision used in stage one of the computation."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor. Dimensions for image cases are `(N x C x H x W)`, where `N` is the batch size, `C` is the number of channels, and `H` and `W` are the height and width of the data. Statistics are computed for every group of channels over `C`, `H`, and `W`. For non-image cases, the dimensions are in the form of `(N x C x D1 x D2 ... Dn)`."
differentiation = "differentiable"

[[inputs]]
name = "scale"
type = "T"
desc = "Scale tensor of shape `(C)`."
differentiation = "differentiable"

[[inputs]]
name = "bias"
type = "T"
desc = "Bias tensor of shape `(C)`."
differentiation = "differentiable"
```

A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each channel. The number of channels
should be divisible by `num_groups` so that there are an equal number of channels
per group.

The overall computation has two stages: the first stage normalizes the elements to
have zero mean and unit variance for each instance in each group, and the second
stage scales and shifts the results of the first stage. The floating-point precision
used in the first stage is determined by the `stash_type` attribute. For example,
if `stash_type` is 1, the operator casts all input variables to 32-bit float,
performs the computation, and finally casts the normalized results back to the
original type of `X`. The second stage does not depend on `stash_type`.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.
