```toml
name = "GroupedMatMul"
domain = "ai.onnx"
node_determinism = "deterministic"

[[inputs]]
name = "input"
type = "T"
desc = "Token matrix with shape [M, K]."
differentiation = "differentiable"

[[inputs]]
name = "weights"
type = "T"
desc = "Stack of G group weight matrices with shape [G, K, N]."
differentiation = "differentiable"

[[inputs]]
name = "group_indices"
type = "Tind"
desc = "Group index for each token and selection, with shape [M, S]. Values outside [0, G) are invalid input."
differentiation = "non_differentiable"

[[inputs]]
name = "bias"
type = "T"
desc = "Optional per-group bias with shape [G, N]."
option = "optional"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Per-group matrix multiplication results with shape [M, S, N]."
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input, weights, bias, and output to floating-point tensors."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain group indices to int32 or int64 tensors."
```

GroupedMatMul multiplies each row of a token matrix by one or more selected
group (expert) weight matrices. Given `input` of shape `[M, K]`, `weights` of
shape `[G, K, N]`, and `group_indices` of shape `[M, S]`, where `S` is the number
of selections per token, the output has shape `[M, S, N]` and is defined by:

```
output[m, i] = input[m] @ weights[group_indices[m, i]]
```

If `bias` is present, `bias[group_indices[m, i]]` is added to each corresponding
result. Every value in `group_indices` must be in the range `[0, G)`.
Values outside this range, including negative indices, are invalid input;
the behavior is undefined for such inputs. In particular, negative indices
do not select groups by counting backwards from `G`.

This operator is represented as a context-dependent function, which also provides
its reference implementation. Its decomposition uses Gather, Expand, and MatMul
to define the result, while runtimes are expected to use a fused
grouped-matrix-multiplication implementation to avoid materializing the expanded
intermediate tensors, including selected weights of shape `[M * S, K, N]`.

Weight-only quantized experts are intended to be expressed by applying
DequantizeLinear to quantized weights (with the appropriate scales and zero
points) and passing the result to GroupedMatMul. Runtimes may fuse this
composition to avoid materializing the dequantized `[G, K, N]` weights.
