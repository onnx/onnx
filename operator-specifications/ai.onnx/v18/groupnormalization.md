```toml
name = "GroupNormalization"
domain = "ai.onnx"
deprecated = true
node_determinism = "deterministic"
shape_inference = false
function_body_builders = { 18 = "defs::nn::utils::GroupNormalizationFunctionBody_opset18" }

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero."
default = 1e-05

[attributes.num_groups]
type = "INT"
desc = "The number of groups of channels. It should be a divisor of the number of channels `C`."
required = true

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor. Dimensions for image cases are `(N x C x H x W)`, where `N` is the batch size, `C` is the number of channels, and `H` and `W` are the height and width of the data. Statistics are computed for every group of channels over `C`, `H`, and `W`. For non-image cases, the dimensions are in the form of `(N x C x D1 x D2 ... Dn)`."
differentiation = "differentiable"

[[inputs]]
name = "scale"
type = "T"
desc = "Scale tensor of shape `(num_groups)`."
differentiation = "differentiable"

[[inputs]]
name = "bias"
type = "T"
desc = "Bias tensor of shape `(num_groups)`."
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "The output tensor of the same shape as `X`."
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```

A GroupNormalization function. Carries out group normalization as described in
the paper https://arxiv.org/abs/1803.08494

This operator transforms input according to
```
y = scale * (x - mean) / sqrt(variance + epsilon) + bias,
```
where the mean and variance are computed per instance per group of channels, and
`scale` and `bias` should be specified for each group of channels. The number of
groups `num_groups` should be divisible by the number of channels so that there are
an equal number of channels per group.

When the number of groups is the same as the number of channels, this operator is
equivalent to InstanceNormalization. When there is only one group, this operator
is equivalent to LayerNormalization.
