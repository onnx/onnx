```toml
name = "RMSNormalization"
domain = "ai.onnx"
node_determinism = "deterministic"
shape_inference = "defs::nn::utils::RMSNormalizationShapeInference_opset23"
function_body_builders = { 23 = "defs::nn::utils::RMSNormalizationFunctionBody_opset23" }

[attributes.axis]
type = "INT"
desc = "The first normalization dimension. If rank(X) is r, axis' allowed range is [-r, r). Negative value means counting dimensions from the back."
default = -1

[attributes.epsilon]
type = "FLOAT"
desc = "The epsilon value to use to avoid division by zero."
default = 1e-05

[attributes.stash_type]
type = "INT"
desc = "The floating-point precision used in stage one of the computation."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "The input tensor to be normalized. In general, the shape is (D1, D2, ... , Dn) for n-dimensional data, where the root mean squared norm is taken over the last D dimensions, D is determined by the axis attribute."

[[inputs]]
name = "scale"
type = "V"
desc = "Scale tensor. Scale tensor shape should be broadcastable to the normalized shape."

[[outputs]]
name = "Y"
type = "V"
desc = "Output data tensor. Same shape as X"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input X type to float tensors."

[type_constraints.V]
types = "all_float_types_ir4"
desc = "Constrain output Y and scale type to float tensors."
```

This is RMS normalization defined in ONNX as function as described in the paper https://arxiv.org/pdf/1910.07467.
The overall computation can be split into two stages. The root mean squared norm is taken over the last D dimensions,
where D is the dimension of normalized_shape. For example, if normalized_shape is (3, 5) (a 2-dimensional shape),
the rms norm is computed over the last 2 dimensions of the input. The computation required by standardization can be
described by the following equations.
```
XSquared = Mul(X, X)
XSquaredMean = ReduceMean<axes=normalized_axes>(XSquared)
MeanSquareEpsilon = Add(XSquaredMean, epsilon)
RMS = Sqrt(MeanSquareEpsilon)
Normalized = Div(X, RMS)
```
where `normalized_axes` is `[axis, ..., rank of X - 1]`. The variables `RMS` stand for root mean square,
Depending on `stash_type` attribute, the actual computation
must happen in different floating-point precision.
For example, if `stash_type` is 1, this operator casts
all input variables to 32-bit float, perform the computation, and
finally cast `Normalized` back to the original type of `X`.
The second stage then scales the outcome of the first stage using:
```
Y= Mul(Normalized, Scale)
```
Let `d[i]` indicate the i-th dimension of `X`.
If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
the shape of `RMS` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
`Y` and `X` have the same shape. This operator supports unidirectional broadcasting
(`Scale` should be unidirectional broadcastable to tensor `X`);
for more details please check [the doc](Broadcasting.md).
