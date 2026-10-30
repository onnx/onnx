<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Composable mixture-of-experts routing

An MoE feed-forward block can be represented with standard ONNX operators:
select a few experts for each token, evaluate only the selected expert
projections, multiply by routing weights, and sum the selected outputs.
Opset 29's `GroupedMatMul` expresses the expert projections without requiring
the graph to gather a separate copy of the expert weights for every route.
The router policy and shared experts remain ordinary graph computations.

The {download}`executable example <../../../examples/moe.py>` constructs two complete
models with reusable model-local functions. It does not register new standard
operators. From a source checkout with ONNX installed, generate the models with:

```shell
python docs/examples/moe.py output
```

`softmax_moe.onnx` contains a softmax top-k router and a sigmoid-gated shared
expert. `grouped_sigmoid_moe.onnx` contains a grouped, bias-corrected sigmoid
router and an ungated shared expert. Both include routing indices and mixture
weights as outputs as well as the final hidden states. Weights and routing
parameters are graph inputs so exporters can replace them with model
initializers. Both models run with `onnx.reference.ReferenceEvaluator`; their
local functions can also be expanded with `onnx.inliner.inline_local_functions`.

## Selected-expert SwiGLU

Flatten the batch and sequence dimensions into `T` tokens before this block.
The example's weight layout is the transpose of the usual framework linear
layer layout:

| Input | Shape | Meaning |
|---|---|---|
| `X` | `[T,H]` | Token hidden states |
| `Gate`, `Up` | `[E,H,I]` | Expert gate and up matrices |
| `Down` | `[E,I,H]` | Expert down matrices |
| `Indices` | `[T,k]` | Selected expert IDs |
| `Weights` | `[T,k]` | Mixture weights, separate from selection scores |

`SelectedSwiGLU` first applies `GroupedMatMul(X, Gate, Indices)` and
`GroupedMatMul(X, Up, Indices)`, producing `[T,k,I]` tensors. The activation is
`SiLU(gate) * up`: sigmoid and multiplication are applied to the gate projection
**before** multiplication by the up projection. Flatten this selected activation
to `[T*k,I]` and the IDs to `[T*k,1]`, then apply `GroupedMatMul` with `Down`.
Reshape to `[T,k,H]`, multiply by `Weights[...,None]`, and `ReduceSum` over `k`.
The result is `[T,H]`. Routing weights are applied after the down projection.

There is no `OneHot[T*k,E]` dispatch tensor, atomic scatter-add requirement, or
graph-level gathering of expert weights. A backend can sort token/expert pairs,
run grouped matrix kernels, undo the permutation, and reduce the routes.
The standard function fallback of `GroupedMatMul` may materialize selected
weights; a native implementation can avoid doing so.
The [Transformers grouped expert integration](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/integrations/moe.py#L392-L505)
illustrates sorting, grouped matrix multiplication, inverse permutation,
and reshape-plus-sum.

## Keep selection separate from mixture weights

`SoftmaxRouter` applies row-wise softmax to `[T,E]` logits, takes `TopK`, and
gathers the selected probabilities. `MixingWeights` optionally divides by the
sum of selected probabilities plus `Epsilon`, then multiplies by `Scale`.
For the Qwen mapping use `Epsilon=0` and `Scale=1`; `Normalize` corresponds to
the model's `norm_topk_prob` setting.

`GroupedRouter` follows these steps:

1. Apply sigmoid to the logits to obtain the **unbiased** `Scores`.
2. Add the `[E]` correction bias to obtain selection-only `Choice` scores.
3. Reshape to `[T,G,P]`, where `GroupShape=[G,P]` and `G*P=E`.
   Rank each group by the sum of its two highest corrected scores.
4. Choose `GroupK` groups. Use `ScatterElements` to form a `[T,G]` mask,
   expand it across the experts within each group, and replace scores outside
   the selected groups with negative infinity.
5. Select `K` experts from the masked corrected scores, but gather the mixture
   weights from the original **unbiased sigmoid scores**.
6. Optionally normalize those weights with `sum + Epsilon`, then multiply by
   `Scale`. The GLM/Kimi source uses `Epsilon=1e-20` and the model's
   `routed_scaling_factor`.

For example, logits `[2,1,0,-1]` in two groups of two ordinarily choose experts
`[0,1]`. Correction `[0,0,3,3]` instead selects `[2,3]`. With no normalization
and scale two, their mixture weights are `[1, 2*sigmoid(-1)]`, **not** the
corrected scores. A correction bias must never become an additive FFN bias or
an additive mixture weight.

## Shared experts

The complete softmax model computes a separate dense SwiGLU shared expert,
multiplies it by `sigmoid(X @ SharedRouter)`, then adds it to the routed output.
The grouped model directly adds its dense shared expert. `J`, the shared
intermediate width, is independent of routed expert width `I`. Multiple
ungated shared experts can be represented by concatenating their gate/up
intermediate channels and concatenating their down matrices along that
intermediate dimension.

## Mapping real models

The following mappings use Transformers source pinned to commit
`c27795950091cdaf1b0e32c4aed58618fb41ffa0`, rather than treating a model name as a
router policy.

| Model | Mapping |
|---|---|
| Qwen3Next | [`Qwen3NextTopKRouter` and `Qwen3NextSparseMoeBlock`](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/qwen3_next/modeling_qwen3_next.py#L782-L860): softmax top-k, optional selected-probability normalization, selected SwiGLU, sigmoid-gated shared expert. |
| Qwen3.8-2.4T-A95B | Its [pinned configuration](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/config.json) selects `Qwen3_5MoeForCausalLM`: `E=512`, `k=10`, `H=8192`, `I=J=2048`. The corresponding [`Qwen3_5Moe` implementation](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L832-L910) maps to the softmax composition. |
| Qwen3.8-Flash-Next | Its [pinned configuration](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) selects `Qwen4ExpForConditionalGeneration`: `E=512`, `k=10`, `H=2560`, `I=J=640`. [`Qwen4Exp` directly reuses the Qwen3Next router, experts and sparse block](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/qwen4_exp/modular_qwen4_exp.py#L511-L528), so the softmax composition represents its MoE FFN. Its [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) describes it as a preview of the architecture underpinning Qwen4. |
| GLM-5 / GLM-MoE-DSA | [`GlmMoeDsaTopkRouter`](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py#L453-L505): grouped sigmoid routing with selection-only correction, sum-of-top-two group ranking, optional normalization and routed scaling. Use the grouped composition plus an ungated shared expert. |
| GLM-4-MoE | [`Glm4MoeTopkRouter`](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/glm4_moe/modeling_glm4_moe.py#L268-L304) uses the same grouped routing pattern. |
| Kimi Linear | [`KimiLinearTopkRouter` and `KimiLinearMoE`](https://github.com/huggingface/transformers/blob/c27795950091cdaf1b0e32c4aed58618fb41ffa0/src/transformers/models/kimi_linear/modeling_kimi_linear.py#L718-L802) directly implement the grouped sigmoid pattern with an ungated shared expert. |

These are mappings of the MoE feed-forward sublayer. Attention, embeddings,
normalization and residual paths remain outside these example graphs.
For Qwen3.8-2.4T-A95B, set `Normalize=true`: its `Qwen3_5MoeTopKRouter`
always normalizes the selected probabilities.

## Input and numerical contract

These examples use **float32 throughout**, including router logits, softmax or
sigmoid, correction, normalization, expert projections and weighted reduction.
They illustrate the floating-point graph, not bitwise equivalence with
low-precision checkpoints or a particular fused kernel. To export a mixed
precision model, preserve its actual cast points explicitly: for example,
Qwen3Next computes softmax in float32 and casts selected weights back, whereas
the GLM/Kimi router also computes its linear projection in float32. Activation
placement and accumulation precision must be preserved when fusing.

Inputs must have compatible shapes and finite values, and router calculations
must produce finite logits and corrected scores. `E`, `H`, `I`, `J` and `k`
are positive. `K` is int64 `[1]` with `1 <= k <= E`; for grouped routing,
`P >= 2`, `1 <= GroupK <= G`, and `k <= GroupK*P`. `Normalize` is a scalar
bool; `Epsilon` is a finite nonnegative float32 scalar; `Scale` is a finite
float32 scalar. Normalization requires a positive denominator; extremely
negative sigmoid logits can underflow to zero, so a positive epsilon is
important for that router. Invalid routing parameters are not clamped.

`TopK` uses descending order and ONNX's lower-index-first rule for equal
scores, both for groups and experts. Framework implementations using unsorted
top-k may choose a different subset on ties. The routers emit distinct expert
IDs for each token. `SelectedSwiGLU` can also be called independently with
duplicate IDs: each occurrence is a separate weighted route and contributes
to the sum. IDs must be in `[0,E)`; no padding/sentinel expert is defined.

Zero tokens (`T=0`) produce `[0,H]`, `[0,k]`, and `[0,k]` outputs. Reshapes
that construct an explicit zero token dimension use `allowzero=1`; no batch
size is inferred by dividing by a zero dimension. Empty expert sets, `k=0`,
and zero hidden/intermediate widths are outside this example's contract.

## Graph construction

The functions and complete graph assembly are shown below. The `local.moe`
domain is example-local and is carried in each serialized model; it does not
require an external custom-operator implementation.

```{literalinclude} ../../../examples/moe.py
:language: python
:start-at: def make_functions
:end-before: if __name__
```
