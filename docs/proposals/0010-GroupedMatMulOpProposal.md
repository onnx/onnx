<!--
Copyright (c) Microsoft Corporation. All rights reserved.
SPDX-License-Identifier: MIT
-->

# RFC: GroupedMatMul — Proposal for ONNX Standard

**Status:** Proposal  
**ONNX Discussion:** *(to be filed as a GitHub Discussion in onnx/onnx)*  
**Related:** [onnx/onnx#7902](https://github.com/onnx/onnx/issues/7902)  
**Reference implementation:** `com.microsoft.GroupedMatMul` (ONNX Runtime `contrib_ops`, opset 1)

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Prior Art and Framework Support](#2-prior-art-and-framework-support)
3. [Semantics (Function Decomposition)](#3-semantics-function-decomposition)
   - 3.1 [Function Decomposition](#31-function-decomposition)
   - 3.2 [Edge Cases](#32-edge-cases)
4. [Operator Specification](#4-operator-specification)
   - 4.1 [Name and Domain](#41-name-and-domain)
   - 4.2 [Inputs](#42-inputs)
   - 4.3 [Outputs](#43-outputs)
   - 4.4 [Type Constraints](#44-type-constraints)
   - 4.5 [Attributes](#45-attributes)
   - 4.6 [Shape Inference Rules](#46-shape-inference-rules)
5. [Test Cases](#5-test-cases)
6. [Typical Usage — MoE Feed-Forward Layer](#6-typical-usage--moe-feed-forward-layer)
7. [Design Alternatives Considered](#7-design-alternatives-considered)

---

## 1. Motivation

Mixture-of-Experts (MoE) feed-forward layers are a central component of large language models (Mixtral, DeepSeek, Grok, Switch Transformer, etc.). The core computation of an MoE layer is:

> Given a batch of `M` token vectors and a set of `num_groups` expert weight matrices, multiply each token by one or more expert matrices chosen per-token by a router. Optionally, combine the `k` per-expert results with learned combine weights.

This pattern — often called **grouped matrix multiplication** or **grouped GEMM** — can be expressed using existing standard ONNX operators, but a straightforward (unfused) implementation will be very inefficient and impractical.

* The natural decomposition (`Gather` weights → `Expand` tokens → batched `MatMul`) materialises a full `[M×k, K, N]` weight slice and an `[M×k, K]` copy of tokens. For real MoE layers these tensors are gigabytes in size, making the decomposition impractical.
* A fused grouped-GEMM kernel, on the other hand, processes each expert weight matrix once regardless of the number of tokens that select it, and reuses each token row across its `k` experts without copying. It can additionally fuse the `k`-way weighted sum (the "combine" step), avoiding a materialised `[M, k, N]` intermediate.


Adding `GroupedMatMul` to the ONNX standard will enable a more compact and efficient representation of MoE models.

---

## 2. Prior Art and Framework Support

| Framework / Library | API |
|---|---|
| PyTorch | `torch.nn.functional.grouped_mm` (PyTorch ≥ 2.5) |
| PyTorch | `torch._grouped_mm` / `torch.ops.aten.mm_group` (internal) |
| JAX | `jax.lax.dot_general` with grouped batching |
| cuBLAS | `cublasGemmBatchedEx` / `cublasGemmGroupedBatchedEx` |
| CUTLASS | `GroupedGemm` kernel |
| OpenVINO | `GroupConvolution` (analogous for convolution) |

The PyTorch `torch.nn.functional.grouped_mm` API (added in 2.5) directly matches the semantics proposed here:

```python
# PyTorch grouped_mm — same semantics
out = torch.nn.functional.grouped_mm(input, weight, offs=None)
# offs are contiguous group offsets; our design uses indices instead (see §7.3)
```

---

## 3. Semantics (Function Decomposition)

This section is the **single, normative definition** of `GroupedMatMul`. The operator is
specified as an ONNX **function**: its meaning is exactly the composition of standard ONNX
operators given below. Because the presence of the optional `combine_weights` and `bias`
inputs changes the graph that is produced, the function body is **context-dependent** (built
via `SetContextDependentFunctionBodyBuilder`, in the style of existing ops such as
`CenterCropPad`).

Defining the semantics as a function has a useful consequence for onnx/onnx: the reference
evaluator (`onnx.reference.ReferenceEvaluator`) automatically executes an operator through
its function body when no dedicated Python kernel is registered, so **no separate reference
implementation file is required**. A single decomposition therefore serves as specification,
documentation, and reference implementation.

Input and output names, shapes, and type constraints are defined in [§4](#4-operator-specification).
The symbols used below are `M`, `K`, `N`, `G` (= `weights.shape[0]`) and `k` (=
`group_indices.shape[1]`).

> **Note on efficiency.** The decomposition materialises a full `[M*k, K, N]` weight slice
> and an `[M*k, K]` copy of the tokens. As explained in [§1](#1-motivation), these
> intermediates are gigabytes in size for real MoE layers, which is precisely why
> `GroupedMatMul` exists as a fused operator: it defines *what* is computed, while runtimes
> are expected to fuse the computation rather than execute the naive decomposition. The
> decomposition is normative for the *result*, not for the *strategy*.

### 3.1 Function Decomposition

```
# Common part — per-expert results r: [M, k, N]
idx_flat  = Reshape(group_indices, [M*k])
W_sel     = Gather(weights, idx_flat, axis=0)            # [M*k, K, N] — duplicates weights!
X         = Reshape(Expand(Unsqueeze(input, 1), [M, k, K]),
                    [M*k, 1, K])                         # [M*k, 1, K] — copies tokens!
r         = Reshape(MatMul(X, W_sel), [M, k, N])         # [M, k, N]

# If `bias` is present, add the per-group bias to each selected expert result:
bias_sel  = Reshape(Gather(bias, idx_flat, axis=0), [M, k, N])   # [M, k, N]
r         = r + bias_sel                                          # (only when bias present)

# If `combine_weights` is present -> weighted sum over the k slots, output [M, N]:
output    = ReduceSum(r * Unsqueeze(combine_weights, -1), axis=1, keepdims=0)   # [M, N]

# Otherwise (combine_weights absent) -> per-expert results, output [M, k, N]:
output    = r                                                                    # [M, k, N]
```

The four resulting cases (with/without `bias` × with/without `combine_weights`) are what the
context-dependent function body emits: the `bias` line is included only when input 4 is
present, and exactly one of the two `output` assignments is emitted depending on whether
input 3 (`combine_weights`) is present.

### 3.2 Edge Cases

The decomposition already gives well-defined behaviour for every special case below; the
table records the resulting behaviour for clarity.

| Case | Behaviour |
|---|---|
| `k == 0` without combine | No expert selected, empty tensor output of shape `[M, 0, N]`. (Not expected in real model.) |
| `k == 0` with combine | No expert selected, output is all zeros of shape `[M, N]`. (Not expected in real model.) |
| `k == 1` without combine | One expert per token, output of shape `[M, 1, N]`. |
| `k == 1` with combine | Effectively scales each token result by `combine_weights[i, 0]`. |
| `G == 1`, all indices 0 | Equivalent to `MatMul(input, weights[0])` (+ optional bias). |
| `M == 0` | Zero-token input; output shape is `[0, N]` or `[0, k, N]`; no compute required. |
| Out-of-range index | Invalid input (implementations must raise an error). |

---

## 4. Operator Specification

### 4.1 Name and Domain

| Field | Value |
|---|---|
| **Name** | `GroupedMatMul` |
| **Domain** | `ai.onnx` (standard) |
| **Opset version** | Next available opset (e.g. 27) |
| **Since version** | (new in this opset) |

### 4.2 Inputs

| Index | Name | Type | Required | Shape | Description |
|---|---|---|---|---|---|
| 0 | `input` | T | **Required** | `[M, K]` | Row-major token matrix. `M` tokens, `K` is the contraction (hidden) dimension. |
| 1 | `weights` | T | **Required** | `[G, K, N]` | Stack of `G` expert weight matrices, each `K × N`. All experts share the same `K` and `N`. |
| 2 | `group_indices` | tensor(int64) | **Required** | `[M, k]` | Group (expert) index per token per slot. Each of the `M` tokens selects `k` experts. Values must be in `[0, G)`. Use `k=1` for the dense (single-expert) case. |
| 3 | `combine_weights` | T | **Optional** | `[M, k]` | Per-selection combine weight. When present, the output is the weighted sum over the `k` selected experts (shape `[M, N]`). When absent, per-expert results are returned (shape `[M, k, N]`). |
| 4 | `bias` | T | **Optional** | `[G, N]` | Per-group bias vector. Added to each expert's result before the optional combine. |

**Notes:**

* `G = weights.shape[0]` (number of groups / experts).
* Callers with batched inputs of shape `[B, M, K]` should `Reshape` the batch dimensions into `M` first. In most backends, typically this is a zero-copy metadata-only view construction.
* `weights` and `bias` are the same for all tokens (i.e., they are model parameters, not per-token).

### 4.3 Outputs

| Index | Name | Type | Shape | Description |
|---|---|---|---|---|
| 0 | `output` | T | `[M, N]` or `[M, k, N]` | When `combine_weights` is present: `[M, N]` (weighted sum over k experts). When absent: `[M, k, N]` (per-expert results). |

### 4.4 Type Constraints

| Constraint | Types |
|---|---|
| `T` | `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)` |

`group_indices` is always `tensor(int64)`.

### 4.5 Attributes

None. All configuration is expressed through inputs (following ONNX's general preference for
inputs over attributes when the values may be dynamic).

### 4.6 Shape Inference Rules

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
5. If `combine_weights` present: `combine_weights.shape == [M, k]`
6. If `bias` present: `bias.shape == [G, N]`

Output shape:
```
output.shape = [M, N]       if combine_weights is present
             = [M, k, N]    otherwise
```

---

## 5. Test Cases

These cases are intended for `onnx/backend/test/case/node/groupedmatmul.py`.

### Test 1 — Dense (k=1), no combine, no bias

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

### Test 2 — Top-k (k=2), no combine, with bias

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

### Test 3 — Top-k (k=2) with combine, no bias

```python
M, k, K, G, N = 2, 2, 2, 3, 2
# (same input/weights/group_indices as Test 2, no bias)
combine_weights = [[0.6, 0.4], [0.3, 0.7]]       # [2, 2]

# token 0: 0.6*[1,0] + 0.4*[0,1] = [0.6, 0.4]
# token 1: 0.3*[0,1] + 0.7*[0,1] = [0.0, 1.0]
output = [[0.6, 0.4], [0.0, 1.0]]                 # [2, 2]
```

### Test 4 — Empty group (one expert unused)

```python
# Group 1 receives no tokens.
M, k, K, G, N = 4, 1, 2, 3, 2
group_indices = [[0], [0], [2], [2]]   # group 1 unused
# weights[1] is never accessed; output is well-defined.
```

### Test 5 — Single group (degenerates to MatMul)

```python
G = 1
group_indices = [[0], [0], [0]]        # all tokens -> group 0
# output == MatMul(input, weights[0]) (reshaped from [M,1,N] to [M,1,N])
```

---

## 6. Typical Usage — MoE Feed-Forward Layer

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

# --- Up projection: per-expert output, no combine ---
# output shape: [B*S, k, F]
h_up = GroupedMatMul(h, expert_gate_W, idx)       # + expert_gate_bias (optional)
h_up = SiLU(h_up)

# Reshape for down projection: treat each (token, expert-slot) pair as a row.
h_flat  = Reshape(h_up, [B*S*k, F])
idx2    = Reshape(idx,  [B*S*k, 1])               # each flat row still in one expert
val2    = Reshape(val,  [B*S*k, 1])               # combine weight (k=1 -> simple scale)

# --- Down projection: fused weighted-sum combine ---
# output shape: [B*S, H]
out = GroupedMatMul(h_flat, expert_down_W, idx2, val2)
out = Reshape(out, [B, S, H])
```

The up-projection uses the **no-combine** form because the activation must run per expert
before the reduce. The down-projection uses the **combine** form to fuse the top-k weighted
sum, avoiding a separate `Mul + ReduceSum` and the round-trip of the `[B*S*k, H]`
intermediate through memory.

---

## 7. Design Alternatives Considered

### 7.1 Special-case shapes for `k==1`

The current proposal requires `group_indices` to be a 2-dimensional tensor of shape `[M, k]`.
An alternative would be to allow a 1-dimensional tensor of shape `[M]` for the `k=1` special case.

### 7.2 Explicit batch dimension
The `M` tokens are flattened into a single dimension in this op. In actual usage, when
batching is used, we might have multi-dimensional tokens, eg, `[Batch, Sequence]`.
We could potentially support an extra batch dimension to avoid extra Reshapes.
We could even let M be any number of dimensions, but that leads to extra complexity that doesn't
seem useful.

### 7.3 Larger Fused Ops

In practice, implementations may use even more aggressive fusions in the implementation
of a MoE feed-forward layer: for example, fusing the down-projection, activation, up-projection, etc.
For example, onnxruntime's contrib op for MoE does this.

The disadvantage is that the activation used varies across models, with new models exploring
use of newer activations all the time. Thus, onnxruntime's contrib op faces a need to
be continuously updated to support newer activations. There is no good solution for this
currently.

**Decision: no fused activation.** Activations stay as separate ONNX ops to keep the graph
composable across the many MoE routing variants. The combine step is (optionally) fused
because it is common to all variants and helps avoid the `Expand`/`ReduceSum` memory overhead.

### 7.3 `group_indices` vs. `group_offsets`

An alternative interface uses a sorted token buffer and integer offsets instead of unsorted
indices. This matches the cuBLAS grouped-GEMM API more directly.

**Decision: `group_indices` (unsorted).** Indices compose naturally with `TopK`/`Gather`
and do not require callers to pre-sort the token batch. Runtimes sort internally.

### 7.4 Stacked 3-D weights `[G, K, N]` vs. a list of variable-size matrices

An alternative would be a sequence of weight tensors with potentially different `K`/`N`
dimensions per expert (the general "heterogeneous-expert" case).

**Decision: stacked `[G, K, N]`.** All experts sharing `K` and `N` is the overwhelmingly
common case in deployed MoE models. A single weight tensor is simpler. (Implementations,
however, may benefit by treating the different experts slices within the single tensor
differently, for example to handle cases where the entire tensor is too large to fit
into memory at same time).


