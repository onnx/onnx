```toml
name = "LayerNormalization"
domain = "ai.onnx"
node_determinism = "deterministic"
allow_unchecked_attributes = true
shape_inference = "defs::nn::utils::LayerNormalizationShapeInference_opset17"
function_body_builders = { 17 = "defs::nn::utils::BuildContextDependentFunctionBodyLayerNormalizationVer17", 18 = "defs::nn::utils::BuildContextDependentFunctionBodyLayerNormalizationVer18" }

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
desc = "Type of Mean and InvStdDev. This also specifies stage one's computation precision."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "Tensor to be normalized."

[[inputs]]
name = "Scale"
type = "T"
desc = "Scale tensor."

[[inputs]]
name = "B"
type = "T"
desc = "Bias tensor."
option = "optional"

[[outputs]]
name = "Y"
type = "T"
desc = "Normalized tensor."

[[outputs]]
name = "Mean"
type = "U"
desc = "Saved mean used during training to speed up gradient computation"
option = "optional"

[[outputs]]
name = "InvStdDev"
type = "U"
desc = "Saved inverse standard deviation used during training to speed up gradient computation."
option = "optional"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input types and output Y type to float tensors."

[type_constraints.U]
types = ["tensor(float)", "tensor(bfloat16)"]
desc = "Type of Mean and InvStdDev tensors."
```

This is layer normalization defined in ONNX as function.
The overall computation can be split into two stages.
The first stage is standardization, which makes the
normalized elements have zero mean and unit variances.
The computation required by standardization can be
described by the following equations.
```
Mean = ReduceMean<axes=normalized_axes>(X)
D = Sub(X, Mean)
DD = Mul(D, D)
Var = ReduceMean<axes=normalized_axes>(DD)
VarEps = Add(Var, epsilon)
StdDev = Sqrt(VarEps)
InvStdDev = Reciprocal(StdDev)
Normalized = Mul(D, InvStdDev)
```
where `normalized_axes` is `[axis, ..., rank of X - 1]`.
The variables `Var` and `StdDev` stand for variance and
standard deviation, respectively. The second output is
`Mean` and the last one is `InvStdDev`.
Depending on `stash_type` attribute, the actual computation
must happen in different floating-point precision.
For example, if `stash_type` is 1, this operator casts
all input variables to 32-bit float, perform the computation, and
finally cast `Normalized` back to the original type of `X`.
The second stage then scales and shifts the outcome of the
first stage using
```
NormalizedScaled = Mul(Normalized, Scale)
Y = Add(NormalizedScaled, B)
```
The second stage doesn't depends on `stash_type`.
All equations are in [this syntax](https://github.com/onnx/onnx/blob/main/docs/Syntax.md).
The same variable (i.e., input, output, and attribute) uses
the same name in the equations above and this operator's definition.
Let `d[i]` indicate the i-th dimension of `X`.
If `X`'s shape is `[d[0], ..., d[axis-1], d[axis], ..., d[rank-1]]`,
the shape of `Mean` and `InvStdDev` is `[d[0], ..., d[axis-1], 1, ..., 1]`.
`Y` and `X` have the same shape. This operator supports unidirectional broadcasting
(tensors `Scale` and `B` should be unidirectional broadcastable to tensor `X`);
for more details please check [the doc](Broadcasting.md).
