<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

- Feature Name: `grouped_matmul`
- Start Date: 2026-07-14
- RFC PR: [onnx/onnx#8193](https://github.com/onnx/onnx/pull/8193)
- Status: under discussion
- Authors:
  - gramalingam

## Summary
[summary]: #summary

This RFC proposes adding a `GroupedMatMul` operator to the standard `ai.onnx` domain.
`GroupedMatMul` multiplies a batch of token vectors by a set of expert weight matrices,
with each token selecting one or more experts, and returns the per-expert results (with an
optional per-expert bias). It provides a compact, efficiently fusable representation of the
core matrix-multiplication computation in Mixture-of-Experts (MoE) feed-forward layers.
The operator is specified as a context-dependent ONNX function, so its meaning is exactly
the composition of standard ONNX operators given in the
[Reference-level explanation](#reference-level-explanation).

ONNX Runtime does not expose a standalone grouped-matmul contrib op; instead it provides a
larger fused `com.microsoft.MoE` (and quantized `com.microsoft.QMoE`) contrib op that
implements an entire MoE feed-forward layer. This proposal is related to
[onnx/onnx#7902](https://github.com/onnx/onnx/issues/7902).

## Motivation
[motivation]: #motivation

Mixture-of-Experts (MoE) feed-forward layers are a central component of large language models (Mixtral, DeepSeek, Grok, Switch Transformer, etc.). The core computation of an MoE layer is:

> Given a batch of `M` token vectors and a set of `num_groups` expert weight matrices, multiply each token by one or more expert matrices chosen per-token by a router.

This pattern — often called **grouped matrix multiplication** or **grouped GEMM** — can be expressed using existing standard ONNX operators, but a straightforward (unfused) implementation will be very inefficient and impractical.

* The natural decomposition (`Gather` weights → `Expand` tokens → batched `MatMul`) materialises a full `[M×k, K, N]` weight slice and an `[M×k, K]` copy of tokens. For real MoE layers these tensors are gigabytes in size, making the decomposition impractical.
* A fused grouped-GEMM kernel, on the other hand, processes each expert weight matrix once regardless of the number of tokens that select it, and reuses each token row across its `k` experts without copying.

Adding `GroupedMatMul` to the ONNX standard will enable a more compact and efficient representation of MoE models.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

`GroupedMatMul` takes a token matrix `input` of shape `[M, K]`, a stack of `G` expert weight
matrices `weights` of shape `[G, K, N]`, and a `group_indices` tensor of shape `[M, k]` that
selects, for each token, the `k` experts it should be multiplied by. It optionally takes a
per-group `bias` (shape `[G, N]`).

The output has shape `[M, k, N]`: the per-expert result for each of the `k` selected experts.
Any weighted combination of these per-expert results (for example the top-`k` router-weighted
sum in an MoE layer) is expressed with standard `Mul` / `ReduceSum` ops in the surrounding
graph (see the example below).

Use `k = 1` for the dense (single-expert) case.

### Typical Usage — MoE Feed-Forward Layer

A standard top-k MoE FFN with two projections maps directly onto two `GroupedMatMul` ops.
No `Expand` of the token batch is required — the op reuses each token row across its `k`
selected experts internally.

```
# Notation: B = batch, S = sequence length, H = hidden dim, F = FFN inner dim
# E = num_experts, k = experts_per_token

scores          = Softmax(MatMul(hidden, router_W))          # [B, S, E]
values, indices = TopK(scores, k)                            # [B, S, k]

h   = Reshape(hidden,  [B*S, H])
idx = Reshape(indices, [B*S, k])
val = Reshape(values,  [B*S, k])

# --- Up projection: per-expert output ---
# output shape: [B*S, k, F]
h_up = GroupedMatMul(h, expert_gate_W, idx)       # + expert_gate_bias (optional)
h_up = SiLU(h_up)

# Reshape for down projection: treat each (token, expert-slot) pair as a row.
h_flat  = Reshape(h_up, [B*S*k, F])
idx2    = Reshape(idx,  [B*S*k, 1])               # each flat row selects one expert

# --- Down projection: per-expert output, then router-weighted sum over the k slots ---
d       = GroupedMatMul(h_flat, expert_down_W, idx2)   # [B*S*k, 1, H]
d       = Reshape(d, [B*S, k, H])                      # regroup the k slots
out     = ReduceSum(d * Unsqueeze(val, -1), axis=1)    # [B*S, H] weighted sum over k
out     = Reshape(out, [B, S, H])
```

Both projections use `GroupedMatMul` purely for the grouped matrix multiplication. The
top-`k` router-weighted sum in the down-projection is expressed explicitly with `Mul` +
`ReduceSum` over the regrouped `k` slots. Fusing that weighted sum into the matmul is a
possible future extension (see [Future possibilities](#future-possibilities)); it is kept
out of this operator because the `k` experts here have distinct per-slot inputs (their
post-activation up-projection outputs) rather than sharing a single input row.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Semantics (Function Decomposition)

This section is the **single, normative definition** of `GroupedMatMul`. The operator is
specified as an ONNX **function**: its meaning is exactly the composition of standard ONNX
operators given below. Because the presence of the optional `bias` input changes the graph
that is produced, the function body is **context-dependent** (built via
`SetContextDependentFunctionBodyBuilder`, in the style of existing ops such as
`CenterCropPad`).

Defining the semantics as a function has a useful consequence for onnx/onnx: the reference
evaluator (`onnx.reference.ReferenceEvaluator`) automatically executes an operator through
its function body when no dedicated Python kernel is registered, so **no separate reference
implementation file is required**. A single decomposition therefore serves as specification,
documentation, and reference implementation.

Input and output names, shapes, and type constraints are defined in
[Operator Specification](#operator-specification).
The symbols used below are `M`, `K`, `N`, `G` (= `weights.shape[0]`) and `k` (=
`group_indices.shape[1]`).

> **Note on efficiency.** The decomposition materialises a full `[M*k, K, N]` weight slice
> and an `[M*k, K]` copy of the tokens. As explained in [Motivation](#motivation), these
> intermediates are gigabytes in size for real MoE layers, which is precisely why
> `GroupedMatMul` exists as a fused operator: it defines *what* is computed, while runtimes
> are expected to fuse the computation rather than execute the naive decomposition. The
> decomposition is normative for the *result*, not for the *strategy*.

#### Function Decomposition

```
idx_flat  = Reshape(group_indices, [M*k])
W_sel     = Gather(weights, idx_flat, axis=0)            # [M*k, K, N] — duplicates weights!
X         = Reshape(Expand(Unsqueeze(input, 1), [M, k, K]),
                    [M*k, 1, K])                         # [M*k, 1, K] — copies tokens!
r         = Reshape(MatMul(X, W_sel), [M, k, N])         # [M, k, N]

# If `bias` is present, add the per-group bias to each selected expert result:
bias_sel  = Reshape(Gather(bias, idx_flat, axis=0), [M, k, N])   # [M, k, N]
r         = r + bias_sel                                          # (only when bias present)

output    = r                                                    # [M, k, N]
```

The two resulting cases (with/without `bias`) are what the context-dependent function body
emits: the `bias` line is included only when input 3 is present.

#### Edge Cases

The decomposition already gives well-defined behaviour for every special case below; the
table records the resulting behaviour for clarity.

| Case | Behaviour |
|---|---|
| `k == 0` | No expert selected, empty tensor output of shape `[M, 0, N]`. (Not expected in real model.) |
| `k == 1` | One expert per token, output of shape `[M, 1, N]`. |
| `G == 1`, all indices 0 | Equivalent to `MatMul(input, weights[0])` (+ optional bias). |
| `M == 0` | Zero-token input; output shape is `[0, k, N]`; no compute required. |
| Out-of-range index | Invalid input (implementations must raise an error). |

### Operator Specification

#### Name and Domain

| Field | Value |
|---|---|
| **Name** | `GroupedMatMul` |
| **Domain** | `ai.onnx` (standard) |
| **Opset version** | Next available opset (e.g. 27) |
| **Since version** | (new in this opset) |

#### Inputs

| Index | Name | Type | Required | Shape | Description |
|---|---|---|---|---|---|
| 0 | `input` | T | **Required** | `[M, K]` | Row-major token matrix. `M` tokens, `K` is the contraction (hidden) dimension. |
| 1 | `weights` | T | **Required** | `[G, K, N]` | Stack of `G` expert weight matrices, each `K × N`. All experts share the same `K` and `N`. |
| 2 | `group_indices` | tensor(int64) | **Required** | `[M, k]` | Group (expert) index per token per slot. Each of the `M` tokens selects `k` experts. Values must be in `[0, G)`. Use `k=1` for the dense (single-expert) case. |
| 3 | `bias` | T | **Optional** | `[G, N]` | Per-group bias vector. Added to each expert's result. |

**Notes:**

* `G = weights.shape[0]` (number of groups / experts).
* Callers with batched inputs of shape `[B, M, K]` should `Reshape` the batch dimensions into `M` first. In most backends, typically this is a zero-copy metadata-only view construction.
* `weights` and `bias` are the same for all tokens (i.e., they are model parameters, not per-token).

#### Outputs

| Index | Name | Type | Shape | Description |
|---|---|---|---|---|
| 0 | `output` | T | `[M, k, N]` | Per-expert results: for each token, the result of multiplying it by each of its `k` selected experts (plus the optional bias). |

#### Type Constraints

| Constraint | Types |
|---|---|
| `T` | `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)` |

`group_indices` is always `tensor(int64)`.

#### Attributes

None. All configuration is expressed through inputs (following ONNX's general preference for
inputs over attributes when the values may be dynamic).

#### Shape Inference Rules

Let:
- `M  = input.shape[0]`
- `K  = input.shape[1]`
- `G  = weights.shape[0]`
- `N  = weights.shape[2]`
- `k  = group_indices.shape[1]`

Validation checks (raise error if violated):
1. `input.rank == 2`
2. `weights.rank == 3`
3. `group_indices.rank == 2` and `group_indices.shape[0] == M`
4. `weights.shape[1] == K` (contraction dimension agrees)
5. If `bias` present: `bias.shape == [G, N]`

Output shape:
```
output.shape = [M, k, N]
```

### Test Cases

These cases are intended for `onnx/backend/test/case/node/groupedmatmul.py`.

#### Test 1 — Dense (k=1), no bias

```python
# 4 tokens, K=3, G=2 groups, N=2, k=1
input          = [[1, 0, -1],
                  [0, 1,  2],
                  [1, 1,  0],
                  [0, 0,  1]]          # shape [4, 3]

weights        = [[[1, 0], [0, 1], [-1, 0]],
                  [[0, 1], [1, 0], [ 0, 1]]]  # shape [2, 3, 2]

group_indices  = [[0], [1], [0], [1]]  # shape [4, 1]

# Expected output shape [4, 1, 2]:
#   token 0 -> group 0: [1,0,-1] @ [[1,0],[0,1],[-1,0]] = [1+0+1, 0+0+0] = [2, 0]
#   token 1 -> group 1: [0,1, 2] @ [[0,1],[1,0],[ 0,1]] = [0+1+0, 0+0+2] = [1, 2]
#   token 2 -> group 0: [1,1, 0] @ [[1,0],[0,1],[-1,0]] = [1+0+0, 0+1+0] = [1, 1]
#   token 3 -> group 1: [0,0, 1] @ [[0,1],[1,0],[ 0,1]] = [0+0+0, 0+0+1] = [0, 1]
output         = [[[2, 0]], [[1, 2]], [[1, 1]], [[0, 1]]]  # shape [4, 1, 2]
```

#### Test 2 — Top-k (k=2), with bias

```python
M, k, K, G, N = 2, 2, 2, 3, 2
input         = [[1.0, 0.0], [0.0, 1.0]]         # [2, 2]
weights       = [[[1,0],[0,1]], [[0,1],[1,0]], [[1,1],[0,0]]]  # [3,2,2]
group_indices = [[0, 1], [2, 0]]                  # [2, 2]
bias          = [[0.1, 0.2], [0.3, 0.0], [0.5, 0.5]]  # [3, 2]

# token 0, slot 0 -> g=0: [1,0] @ [[1,0],[0,1]] + [0.1,0.2] = [1.1, 0.2]
# token 0, slot 1 -> g=1: [1,0] @ [[0,1],[1,0]] + [0.3,0.0] = [0.3, 1.0]
# token 1, slot 0 -> g=2: [0,1] @ [[1,1],[0,0]] + [0.5,0.5] = [0.5, 0.5]
# token 1, slot 1 -> g=0: [0,1] @ [[1,0],[0,1]] + [0.1,0.2] = [0.1, 1.2]
output = [[[1.1, 0.2], [0.3, 1.0]],
          [[0.5, 0.5], [0.1, 1.2]]]   # [2, 2, 2]
```

#### Test 3 — Empty group (one expert unused)

```python
# Group 1 receives no tokens.
M, k, K, G, N = 4, 1, 2, 3, 2
group_indices = [[0], [0], [2], [2]]   # group 1 unused
# weights[1] is never accessed; output is well-defined.
```

#### Test 4 — Single group (degenerates to MatMul)

```python
G = 1
group_indices = [[0], [0], [0]]        # all tokens -> group 0
# output == MatMul(input, weights[0]) (reshaped from [M,1,N] to [M,1,N])
```

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

**Why this design?** Expressing `GroupedMatMul` as a context-dependent ONNX function gives a
single normative definition that doubles as the reference implementation, keeps the operator
composable with the surrounding graph (router, activation, reshapes, weighted sum), and lets
runtimes fuse the computation without prescribing a fusion strategy. Restricting the operator
to the grouped matrix multiplication itself (with an optional per-group `bias`) keeps it small
and general; the router-weighted sum used in MoE layers is left to standard `Mul` / `ReduceSum`
ops, and can be addressed by a separate fused operator later (see
[Future possibilities](#future-possibilities)).

**Impact of not doing this.** Without `GroupedMatMul`, MoE models must either be exported with
the naive `Gather`/`Expand`/`MatMul` decomposition (which materialises gigabyte-scale
intermediates and is impractical), or vendor-specific contrib ops such as ONNX Runtime's `com.microsoft.MoE`, which are not
portable across the ONNX ecosystem.

The following alternatives were considered.

### Special-case shapes for `k==1`

The current proposal requires `group_indices` to be a 2-dimensional tensor of shape `[M, k]`.
An alternative would be to allow a 1-dimensional tensor of shape `[M]` for the `k=1` special case.

### Explicit batch dimension

The `M` tokens are flattened into a single dimension in this op. In actual usage, when
batching is used, we might have multi-dimensional tokens, eg, `[Batch, Sequence]`.
We could potentially support an extra batch dimension to avoid extra Reshapes.
We could even let M be any number of dimensions, but that leads to extra complexity that doesn't
seem useful.

### Larger Fused Ops

In practice, implementations may use even more aggressive fusions in the implementation
of a MoE feed-forward layer: for example, fusing the down-projection, activation, up-projection, etc.
For example, onnxruntime's contrib op for MoE does this.

The disadvantage is that the activation used varies across models, with new models exploring
use of newer activations all the time. Thus, onnxruntime's contrib op faces a need to
be continuously updated to support newer activations. There is no good solution for this
currently.

**Decision: no fused activation, and no fused combine in this operator.** Activations stay as
separate ONNX ops to keep the graph composable across the many MoE routing variants. The
router-weighted sum ("combine") is likewise expressed with standard `Mul` / `ReduceSum` rather
than folded into `GroupedMatMul`: in a standard MoE down-projection the `k` experts have
distinct per-slot inputs, so the weighted sum reduces over a *different* grouping than the
matmul's own `k`, and folding it in would require decoupling the two groupings and pinning down
a normative result layout — added specification and shape-inference complexity for a fusion
that a dedicated future operator can capture more cleanly (see
[Future possibilities](#future-possibilities)).

### `group_indices` vs. `group_offsets`

An alternative interface uses a sorted token buffer and integer offsets instead of unsorted
indices. This matches the cuBLAS grouped-GEMM API more directly.

**Decision: `group_indices` (unsorted).** Indices compose naturally with `TopK`/`Gather`
and do not require callers to pre-sort the token batch. Runtimes sort internally.

### Stacked 3-D weights `[G, K, N]` vs. a list of variable-size matrices

An alternative would be a sequence of weight tensors with potentially different `K`/`N`
dimensions per expert (the general "heterogeneous-expert" case).

**Decision: stacked `[G, K, N]`.** All experts sharing `K` and `N` is the overwhelmingly
common case in deployed MoE models. A single weight tensor is simpler. (Implementations,
however, may benefit by treating the different experts slices within the single tensor
differently, for example to handle cases where the entire tensor is too large to fit
into memory at same time).

## Prior art
[prior-art]: #prior-art

| Framework / Library | API |
|---|---|
| PyTorch | `torch.nn.functional.grouped_mm` (PyTorch ≥ 2.5) |
| PyTorch | `torch._grouped_mm` / `torch.ops.aten.mm_group` (internal) |
| JAX | `jax.lax.dot_general` with grouped batching |
| cuBLAS | `cublasGemmBatchedEx` / `cublasGemmGroupedBatchedEx` |
| CUTLASS | `GroupedGemm` kernel |
| OpenVINO | `GroupConvolution` (analogous for convolution) |
| ONNX Runtime | `com.microsoft.MoE` / `com.microsoft.QMoE` (`contrib_ops`) — a larger fused MoE layer, not a standalone grouped matmul |

The PyTorch `torch.nn.functional.grouped_mm` API (added in 2.5) directly matches the semantics proposed here:

```python
# PyTorch grouped_mm — same semantics
out = torch.nn.functional.grouped_mm(input, weight, offs=None)
# offs are contiguous group offsets; our design uses indices instead
# (see "group_indices vs. group_offsets" under Rationale and alternatives)
```

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- Should the `k == 1` case be allowed a 1-D `group_indices` of shape `[M]`, or should the
  operator always require the 2-D `[M, k]` form? (See *Special-case shapes for `k==1`*.)
- Should the operator take an explicit batch dimension (e.g. multi-dimensional token inputs
  such as `[B, S, K]`) to avoid the surrounding `Reshape`s, or keep the flattened `[M, K]`
  form? (See *Explicit batch dimension* under Rationale and alternatives.)

## Future possibilities
[future-possibilities]: #future-possibilities

- **Fused down-projection with reduction.** A separate operator could represent an MoE
  down-projection: a `GroupedMatMul` whose own `k` is 1 (each flattened `(token, slot)` row
  selects a single expert) fused with the subsequent router-weighted `ReduceSum` over a
  *different* group size `k > 1` (the `k` experts of each token). This fuses the weighted sum
  that the current proposal leaves as explicit `Mul` / `ReduceSum`, at the cost of decoupling
  the matmul grouping from the reduction grouping (with a total-count constraint such as
  `M*k == M2*k2`) and defining a normative result layout. It is deferred to a follow-up
  proposal rather than complicating `GroupedMatMul`.
- **Heterogeneous experts.** Support for experts with differing `K`/`N` via a sequence of
  weight tensors, covering the general grouped-GEMM case.
- **Additional fused steps.** More aggressive fusion (activation, up-/down-projection) as
  seen in some runtime contrib ops, if the ecosystem converges on a stable set of activations.
- **`group_offsets` variant.** A sorted-buffer/offsets interface could be added later for
  backends that map more naturally onto the cuBLAS grouped-GEMM API.
