<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

- Feature Name: `linear_attention_and_causal_conv`
- Start Date: 2026-03-26
- RFC PR: [onnx/onnx#7767](https://github.com/onnx/onnx/pull/7767)
- Status: under discussion
- Authors:
  - @justinchuby

## Summary
[summary]: #summary

This proposal introduces two new ONNX operators — `LinearAttention` and `CausalConvWithState`
— to enable efficient representation of linear attention and state-space model (SSM) layers
that are now present in production LLMs (Qwen3.5, Jamba, RWKV-6, FalconMamba). These operators
fill a critical gap: ONNX currently has no way to represent the recurrent state-update pattern
shared by the entire family of linear attention variants, which means ONNX models cannot
benefit from the 10–50× speedup available through fused GPU kernels. Discussion thread:
[onnx/onnx#7689](https://github.com/onnx/onnx/issues/7689).

## Motivation
[motivation]: #motivation

### A new generation of hybrid LLM architectures

A new generation of hybrid architectures is rapidly replacing pure softmax-attention Transformers
as the state of the art for large language models. These models interleave traditional softmax
attention layers with **linear attention** (or SSM) layers that use a fixed-size recurrent state
instead of a growing KV cache:

| Model | Release | Linear Attention Type | Hybrid? |
|-------|---------|----------------------|---------|
| **Qwen3.5** (Alibaba) | 2025–2026 | Gated DeltaNet | Yes — ~75% linear, ~25% softmax |
| **Jamba** (AI21) | 2024 | Mamba selective scan | Yes — interleaved SSM + attention |
| **RWKV-6** | 2024 | Channel-wise gated linear | No — 100% linear |
| **Mamba-2** (Tri Dao et al.) | 2024 | Structured state space dual (SSD) | No — 100% SSM |
| **GLA** (Yang et al.) | 2024 | Gated linear attention | No — 100% linear |
| **FalconMamba** (TII) | 2024 | Mamba selective scan | No — 100% SSM |

These models share a key property: during autoregressive decoding, each linear attention layer
maintains a **fixed-size state matrix** $S \in \mathbb{R}^{d_k \times d_v}$ that is updated in
$O(d^2)$ time per token, with $O(1)$ memory. This is fundamentally different from softmax
attention, which requires a KV cache that grows linearly with sequence length.

**ONNX currently has no way to represent these operations efficiently.** All 12+ attention-related
operators in ONNX standard and ORT contrib domains are exclusively softmax-based. The ongoing
FlexAttention proposals ([#7494](https://github.com/onnx/onnx/issues/7494),
[#6389](https://github.com/onnx/onnx/issues/6389)) extend softmax attention with custom
score/mask modifications — they do not address linear attention at all.

### The performance problem

We have implemented both Gated DeltaNet and Mamba selective scan by decomposing them into
primitive ONNX ops in the [onnxruntime/mobius](https://github.com/onnxruntime/mobius) project.
This works for **correctness** — the graphs produce numerically equivalent results — but not for
**performance**:

- The Gated DeltaNet recurrent step decomposes into **~20 individual ONNX ops** (Exp, Mul, MatMul,
  Sub, Add, Unsqueeze, Squeeze, Transpose, ReduceSum, etc.)
- Each op requires a separate GPU kernel launch and materializes intermediate tensors to global memory
- The state matrix has shape $(B, H, 128, 128)$. Each $(128, 128)$ slice per $(\text{batch}, \text{head})$
  is $128 \times 128 \times 2$ bytes ≈ **32 KB in fp16**; for a representative batch size $B = 16$
  this is $32\ \text{KB} \times 16 = \mathbf{512\ \text{KB}}$ of state per head across the batch. The decomposed path reads
  and writes this to global memory **5+ times per token** (decay, retrieve, delta, update, read).
  A fused kernel keeps it in registers/shared memory for the entire update.
- **Estimated performance gap: 10–50×** slower than fused kernels, driven by memory bandwidth
- For a model like Qwen3.5 where **75% of layers** are Gated DeltaNet, this overhead dominates
  total inference time

For a Qwen3.5-9B model with 24 linear attention layers generating 1000 tokens:
- **Fused**: 24 × 1000 × 1 = 24,000 kernel launches for linear layers
- **Decomposed**: 24 × 1000 × ~20 = 480,000 kernel launches — **20× more**

Without dedicated operators, ONNX risks becoming unable to competitively serve the next generation
of LLMs.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

### Background: linear attention family

All linear attention variants share the same structural pattern:

$$S_t = f(S_{t-1}, k_t, v_t, \text{gate}_t) \qquad o_t = q_t^\top S_t$$

where $S \in \mathbb{R}^{d_k \times d_v}$ is a fixed-size state matrix per head. The variants
differ only in the update rule $f$:

| Variant | Update Rule | Models |
|---------|-------------|--------|
| Linear attention | $S_t = S_{t-1} + k_t \otimes v_t$ | Vanilla linear attention |
| RetNet / GLA | $S_t = e^{g_t} S_{t-1} + k_t \otimes v_t$ | RetNet, GLA, RWKV-6 |
| DeltaNet | $S_t = S_{t-1} + \beta_t k_t \otimes (v_t - S_{t-1}^\top k_t)$ | DeltaNet |
| Gated DeltaNet | $S_t = e^{g_t} S_{t-1} + \beta_t k_t \otimes (v_t - e^{g_t} S_{t-1}^\top k_t)$ | **Qwen3.5** |

These models operate in two modes during inference:
1. **Decode (T=1)**: process one token at a time, update the fixed-size state.
2. **Prefill (T>1)**: process a long sequence via chunk-parallel decomposition.

Both modes use identical tensor layouts, enabling a **single unified operator**.

### Using `LinearAttention`

The `LinearAttention` operator accepts Q, K, V plus an optional recurrent `past_state` and
returns the attention output alongside the updated state:

```python
# Decode: one token at a time
output, state = LinearAttention(q, k, v, past_state, decay, beta,
                                update_rule="gated_delta", scale=0.0,
                                q_num_heads=16, kv_num_heads=16)

# Prefill: full sequence (same op, T > 1)
output, final_state = LinearAttention(q, k, v, initial_state, decay, beta,
                                      update_rule="gated_delta", scale=0.0,
                                      q_num_heads=16, kv_num_heads=16)

# Packed QKV: k and v omitted, q contains concatenated QKV
output, state = LinearAttention(packed_qkv, None, None, past_state, decay, beta,
                                update_rule="gated_delta",
                                q_num_heads=16, kv_num_heads=16)
```

The operator supports **3D packed inputs** `[B, T, H*D]`; `q_num_heads` and `kv_num_heads`
are always required. The op internally unpacks to 4D for computation.

### Using `CausalConvWithState`

Stateful causal depthwise convolution used as the preprocessing step in Gated DeltaNet and Mamba:

```python
output, carry_state = CausalConvWithState(input, weight, bias, past_state,
                                          ndim=1, activation="silu")
```

The `ndim` attribute generalizes the op to 1D, 2D, or 3D spatial dimensions.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Operator 1: `LinearAttention`

Unified linear attention operator that handles both autoregressive decoding (T=1) and prefill
(T>1) through a single interface. All inputs use **3D packed format** `[B, T, H*D]`;
`q_num_heads` and `kv_num_heads` are always required. The op internally unpacks to 4D for
computation.

#### Inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `query` | T | $(B, T, H_q \cdot d_k)$ | Packed query vectors. When `key`/`value` are absent, contains packed QKV: $(B, T, (H_q + 2H_{kv}) \cdot d_k)$ |
| `key` | T (optional) | $(B, T, H_{kv} \cdot d_k)$ | Packed key vectors. If absent, `query` must contain packed QKV. |
| `value` | T (optional) | $(B, T, H_{kv} \cdot d_v)$ | Packed value vectors. If absent, `query` must contain packed QKV. |
| `past_state` | S (optional) | $(B, H_{kv}, d_k, d_v)$ | Recurrent state from previous step or context (always 4D). If omitted, the operator **MUST** behave as if a zero-initialized tensor of this shape were provided, representing the first token/chunk with no prior context. |
| `decay` | T (optional) | $(B, T, H_{kv} \cdot d_k)$ or $(B, T, H_{kv})$ | Packed decay gates (log-space). Broadcastable: `(B, T, H_{kv})` for per-head scalar (DeltaNet/RetNet), `(B, T, H_{kv} * d_k)` for per-key-dimension (GLA/RWKV-6). Required for `gated_*` modes. |
| `beta` | T (optional) | $(B, T, H_{kv})$ or $(B, T, 1)$ | Packed update rates (sigmoid output). Broadcastable. Required for `*_delta` modes. |

The op internally reshapes `[B, T, H*D]` → `[B, T, H, D]` → transposes to `[B, H, T, D]`,
computes, then transposes and reshapes the output back to 3D.

#### Packed QKV

When `key` and `value` are both absent, `query` contains all three tensors concatenated along the
packed head dimension. The op splits using `q_num_heads` and `kv_num_heads` (always required):

- `query` shape is $(B, T, (H_q + 2H_{kv}) \cdot d_k)$.
- The op unpacks to `[B, T, H_q + 2H_{kv}, d_k]`, then splits: Q = first $H_q$ heads, K = next $H_{kv}$ heads, V = last $H_{kv}$ heads.

`key` and `value` must either both be provided or both be absent.

#### Outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `output` | T | $(B, T, H_q \cdot d_v)$ | Attention output (3D packed) |
| `present_state` | S | $(B, H_{kv}, d_k, d_v)$ | Updated recurrent state (always 4D) |

**Type parameters:**
- `T`: activation dtype for `query`, `key`, `value`, `decay`, `beta`, and `output` (float16, bfloat16, or float32).
- `S`: state dtype for `past_state` and `present_state`. Must be float32 or the same as `T`. Using `S = float32` with `T = float16/bfloat16` is the recommended configuration for long sequences; runtimes handle any necessary casting internally.

#### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `q_num_heads` | int | — | Number of query heads. **Always required.** |
| `kv_num_heads` | int | — | Number of key/value heads. **Always required.** |
| `update_rule` | string | `"gated_delta"` | One of: `"linear"`, `"gated"`, `"delta"`, `"gated_delta"` |
| `scale` | float | `0.0` | Output scaling factor. When `0.0` (default), the op derives the per-head key dimension $d_k$ from the `query` shape and `q_num_heads`: (1) if `key`/`value` are present (Q-only mode), $d_k = \text{query.shape}[-1] / \text{q\_num\_heads}$; (2) if `key`/`value` are absent (packed-QKV mode), $d_k = \text{query.shape}[-1] / (\text{q\_num\_heads} + 2 \cdot \text{kv\_num\_heads})$. In both cases $d_k$ must be an integer, and the op uses $1/\sqrt{d_k}$. Set explicitly to override (e.g., `scale=1.0` when queries are pre-scaled by the caller). |
| `chunk_size` | int | `64` | Chunk size for the chunk-parallel WY decomposition during prefill (T>1). Tuning hint; does not affect output correctness. |

#### Mathematical definition by `update_rule`

All modes share the same state shape $S \in \mathbb{R}^{B \times H \times d_k \times d_v}$.
Let $s = \text{scale}$ if `scale != 0.0`, else $s = 1/\sqrt{d_k}$.

- **`"linear"`** (vanilla linear attention):
  $$S_t = S_{t-1} + k_t \otimes v_t \qquad o_t = s \cdot q_t^\top S_t$$

- **`"gated"`** (RetNet / GLA style, requires `decay`):
  $$S_t = e^{g_t} \cdot S_{t-1} + k_t \otimes v_t \qquad o_t = s \cdot q_t^\top S_t$$

- **`"delta"`** (DeltaNet, requires `beta`):
  $$S_t = S_{t-1} + \beta_t \cdot k_t \otimes (v_t - S_{t-1}^\top k_t) \qquad o_t = s \cdot q_t^\top S_t$$

- **`"gated_delta"`** (Gated DeltaNet — Qwen3.5, requires `decay` and `beta`):
  $$S_t = e^{g_t} \cdot S_{t-1} + \beta_t \cdot k_t \otimes (v_t - e^{g_t} \cdot S_{t-1}^\top k_t) \qquad o_t = s \cdot q_t^\top S_t$$

where $g_t$ is the `decay` input at position $t$.

#### Reference pseudocode (`gated_delta`, internal 4D computation)

```python
def linear_attention(query_3d, key_3d, v_3d, past_state, decay_3d, beta_3d, scale=0.0,
                     update_rule="gated_delta"):
    # Inputs are 3D: query_3d (B, T, H_q*d_k), key_3d (B, T, H_kv*d_k), etc.
    # Op unpacks to 4D internally before computation:
    q, k, v = unpack_3d(query_3d, key_3d, v_3d, q_num_heads, kv_num_heads)
    decay = unpack_3d_decay(decay_3d, kv_num_heads)
    beta = unpack_3d_beta(beta_3d, kv_num_heads)
    # q, k: (B, H, T, d_k), v: (B, H, T, d_v)
    # decay: (B, H, T, d_k) or broadcastable, beta: (B, H, T, 1) or broadcastable
    d_k = q.shape[-1]
    s = scale if scale != 0.0 else (1.0 / sqrt(d_k))
    T = q.shape[2]
    state = past_state  # (B, H, d_k, d_v)

    outputs = []
    for t in range(T):
        qt = q[:, :, t, :]              # (B, H, d_k)
        kt = k[:, :, t, :]              # (B, H, d_k)
        vt = v[:, :, t, :]              # (B, H, d_v)
        gt = exp(decay[:, :, t, :])     # (B, H, d_k) — broadcastable
        bt = beta[:, :, t, :]           # (B, H, 1) — broadcastable

        state = gt.unsqueeze(-1) * state                       # apply decay
        retrieved = einsum('bhkv,bhk->bhv', state, kt)        # read from memory
        delta = bt * (vt - retrieved)                          # error correction
        state = state + einsum('bhk,bhv->bhkv', kt, delta)    # write to memory
        outputs.append(s * einsum('bhkv,bhk->bhv', state, qt))  # query

    output = stack(outputs, dim=2)  # (B, H, T, d_v), repacked to (B, T, H*d_v) on return
    return output, state
```

For T>1 (prefill), the runtime is expected to use a more efficient **chunk-parallel WY
decomposition** rather than the sequential loop above. The sequential formulation defines the
correct output; the chunk-parallel algorithm achieves the same result with better GPU parallelism.

#### Architecture mapping

| `update_rule` | Required optional inputs | Models |
|---------------|--------------------------|--------|
| `"linear"` | — | Vanilla linear attention |
| `"gated"` | `decay` | RetNet, GLA, RWKV-6 |
| `"delta"` | `beta` | DeltaNet |
| `"gated_delta"` | `decay`, `beta` | **Qwen3.5**, Qwen3-Next, Gated DeltaNet |

#### State precision (`stash_type` convention)

Following the ONNX `stash_type` convention (as used in `LayerNormalization` and `GroupNormalization`),
the type parameter `S` controls the precision of the recurrent state tensors:

The op's externally visible activations (`query`, `key`, `value`, `output`) use the **native dtype
`T`** (float16, bfloat16, or float32) — there is no requirement for callers to upcast these to
float32 at the API boundary.

The recurrent state (`past_state`/`present_state`) uses dtype `S`, which may be float32 even when
`T` is float16 or bfloat16. Runtimes that use float32 state accumulation handle any necessary
casting between `S` and `T` internally, without changing the public tensor dtypes seen by the
model. This matches LSTM's convention of maintaining a float32 cell state regardless of the
activation dtype.

---

### Operator 2: `CausalConvWithState`

Stateful causal depthwise convolution, generalized to N spatial dimensions. Used as the
preprocessing step in Gated DeltaNet and Mamba architectures.

#### Inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `input` | T | $(B, C, ...)$ | Input tensor. Spatial dims: 1D: $(L,)$; 2D: $(H, W)$; 3D: $(D, H, W)$ |
| `weight` | T | $(C, 1, k_1, ...)$ | Depthwise convolution kernel (spatial kernel sizes: $(k_1, \ldots, k_{\text{ndim}})$) |
| `bias` | T (optional) | $(C,)$ | Per-channel bias |
| `past_state` | T (optional) | ndim=1: $(B, C, k_1{-}1)$; ndim=2: $(B, C, H, k_2{-}1)$; ndim=3: $(B, C, D, H, k_3{-}1)$ | Carry-over state (last $k-1$ positions along the causal axis). If omitted, treated as zeros. |

#### Outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `output` | T | Same as `input` | Convolution output after optional activation |
| `present_state` | T | Same shape as `past_state` | Updated carry state for next call |

#### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ndim` | int | `1` | Spatial dimensionality: `1`, `2`, or `3` |
| `activation` | string | `"none"` | Optional fused activation: `"silu"`, `"swish"`, `"none"`. (`"silu"` and `"swish"` are aliases.) |

Causality is always enforced on the **last spatial dimension** (position axis). For ndim=1 this
is $L$; for ndim=2 it is $W$; for ndim=3 it is $W$. All other spatial dimensions ($H$, $D$) are
treated as independent channels — not causal. Causal padding (size $k-1$) is applied on the
left (past-facing) side of the last spatial dimension only, ensuring the output at position $t$
depends only on positions $\leq t$ along that axis.

---

### Complement to existing softmax attention ops

```
Existing ONNX attention ops          Proposed linear attention ops
─────────────────────────────         ──────────────────────────────
• Softmax-based                       • Recurrent state-based
• O(n²) compute, O(n) cache          • O(d²) compute, O(1) cache
• Full pairwise attention             • Fixed-size memory read/write
• For standard Transformer layers     • For linear/SSM layers
                                      • Same model often uses BOTH
```

In hybrid models like Qwen3.5, **both** operator families are needed — softmax attention for the
~25% full attention layers, and `LinearAttention` for the ~75% Gated DeltaNet layers.

## Drawbacks
[drawbacks]: #drawbacks

- **Operator proliferation**: Adding two new operators increases the surface area of the ONNX
  standard. However, both operators cover a well-defined, widely-used pattern with clear precedent
  from the `Attention` op.

- **Implementation burden**: Runtime backends must implement fused kernels to realize the
  performance benefits. The decomposed fallback (using existing ONNX ops) always exists but
  provides no speedup. The Phase 1 contrib approach mitigates this — runtimes can implement
  selectively.

- **`update_rule` extensibility**: Future linear attention variants may not fit cleanly into the
  four proposed update rules. However, new string values can be added to the enum without schema
  changes.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

### Why a single `LinearAttention` op?

An earlier draft proposed two separate operators (`LinearAttentionRecurrent` for decode and
`LinearAttentionChunkParallel` for prefill). The unified design is better because:

1. **Identical I/O contract**: Both modes use the same tensor shapes (T=1 vs T>1 is the only
   difference), the same state dimensions, and the same attributes.
2. **Mirrors ONNX `Attention`**: The `Attention` op handles both prefill and decode (with/without
   KV cache) through a single interface. `LinearAttention` follows the same pattern.
3. **Simpler graphs**: No need to choose between ops at graph construction time or swap them at
   runtime. A single node works for both decode loops and prefill passes.
4. **Runtime freedom**: The runtime can dispatch T=1 to an optimized recurrent kernel and T>1 to
   chunk-parallel internally. The `chunk_size` attribute is a tuning hint, not a correctness param.

### Why 3D-only inputs?

`LinearAttention` accepts only 3D packed inputs `[B, T, H*D]` with `q_num_heads` and
`kv_num_heads` always required. This is a deliberate simplification:

1. **Reduced implementation complexity**: Supporting one input layout halves the kernel variants
   backends must implement. The 3D → 4D unpacking is a trivial reshape+transpose that the runtime
   can fuse with the first computation step.
2. **Alignment with existing linear attention kernels**: The `com.microsoft.LinearAttention` contrib
   op and all major fused kernel libraries (flash-linear-attention, etc.) use the packed
   `[B, T, H*D]` layout. Aligning with this convention reduces porting friction.
3. **No ambiguity**: Graph builders always know they need to pack heads before calling
   `LinearAttention`. There is no layout selection decision at graph construction time.

### Why broadcastable packed decay?

In the 3D packed convention, decay uses shapes `(B, T, H_{kv})` for per-head scalar decay
(DeltaNet/RetNet) or `(B, T, H_{kv} \cdot d_k)` for per-key-dimension decay (GLA/RWKV-6).
Internally (after unpacking), these correspond to `(B, H, T, 1)` and `(B, H, T, d_k)`. The
packed broadcastable design supports both:

- **Per-head scalar decay**: `(B, T, H_{kv})` — one scalar gate per head per token
- **Per-key-dimension decay**: `(B, T, H_{kv} \cdot d_k)` — independent gate per key dimension

A per-key-dimension decay allows the model to selectively retain or forget different components of
the state matrix independently — a key capability of GLA and RWKV-6 that per-head scalar cannot
express.

### Why packed QKV?

Many transformer implementations fuse Q, K, V projections into a single weight matrix, producing
a packed QKV tensor. Making `key` and `value` optional allows the op to accept the fused projection
output directly, avoiding caller-side splits that add memory traffic. This mirrors the behavior of
the ORT `com.microsoft.Attention` contrib op.

### Why `update_rule` as string?

String values (not integers) are forward-compatible: new variants can be added by defining a new
string value without any schema change. This is particularly important given how rapidly new linear
attention variants are being published.

### Alternatives not pursued

- **FlexAttention extension**: FlexAttention proposals modify softmax attention with custom
  score/mask functions. They cannot represent the recurrent state update that is fundamental to
  linear attention.
- **Scan-based ONNX decomposition**: The existing `Scan` op can express the recurrence, but the
  resulting graphs are large (~70 ops for Gated DeltaNet), prevent fusion, and perform poorly in
  practice. We have confirmed this through the onnxruntime/mobius implementation.

## Prior art
[prior-art]: #prior-art

### Fused kernel libraries

The PyTorch ecosystem has production-quality fused kernels for all proposed variants:

- **[flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)** (FLA) —
  Triton kernels for GLA, DeltaNet, Gated DeltaNet, RetNet, RWKV, and more. Used by HuggingFace
  transformers for Qwen3.5 inference.
- **[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)** — CUDA kernel for causal 1D
  convolution with state. Used by Mamba and Gated DeltaNet.
- **[mamba-ssm](https://github.com/state-spaces/mamba)** — CUDA kernels for selective scan.

### ONNX precedent

The ONNX `Attention` op (opset 23) provides direct precedent for this proposal:
- It started as `com.microsoft.Attention` in ORT contrib, was validated in production, and was
  promoted to the standard opset.
- It handles GQA via `q_num_heads`/`kv_num_heads` and accepts optional past KV and attn_mask —
  design patterns carried forward in this proposal (with `LinearAttention` using 3D-only inputs).

### Existing ONNX workaround

The current workaround (used in [onnxruntime/mobius](https://github.com/onnxruntime/mobius)) is
decomposition into ~70 primitive ONNX ops using `Scan`. This produces correct results but is
10–50× slower than fused implementations due to memory bandwidth bottlenecks.

## Discussion: Runtime Implementation Considerations
[runtime-implementation]: #runtime-implementation

The following addresses the question: *"Can backends actually implement this efficiently?"*
The short answer is **yes** — the recurrent decode kernel is comparable in complexity to LSTM,
while the chunk-parallel prefill is more complex but follows established patterns from the FLA
library. This section covers GPU, CPU, NPU, compiler backends, and the fallback strategy.

### 1. GPU Kernel: Fused vs. Decomposed — Over an Order of Magnitude Gap

The [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (FLA) library
provides production Triton kernels used by HuggingFace transformers for Qwen3.5 inference today.

**Fused recurrent kernel (decode)**: One Triton program per (batch, head, value\_tile). The state
tile `[BK=64, BV=64]` lives entirely in **fp32 registers** — never touches global memory during
the token loop:

```python
# Simplified from fla/ops/gated_delta_rule/fused_recurrent.py
b_h = tl.zeros([BK, BV], dtype=tl.float32)   # State in registers
for _ in range(T):
    b_q, b_k, b_v, b_beta, b_g = load_token(...)
    b_h *= exp(b_g)                            # Decay
    b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))  # Delta
    b_h += b_k[:, None] * b_v                  # Update
    b_o = tl.sum(b_h * b_q[:, None], 0)        # Readout
    store(b_o)
```

**Roofline analysis** for one head, $d_k = d_v = 128$:

| | FLOPs | Memory Traffic | Arithmetic Intensity | A100 Position |
|-|-------|---------------|---------------------|---------------|
| **Fused kernel** | ~115K | ~1.3 KB (Q,K,V,g,β,output) | **~90 FLOPs/byte** | ✅ Compute-bound |
| **Decomposed (~20 ONNX ops)** | ~115K | ~640 KB (state round-trips 5× read+write) | **~0.18 FLOPs/byte** | ❌ Memory-bound |

The A100's compute/bandwidth ridge point is ~10 FLOPs/byte. The fused kernel sits at ~90
(deeply compute-bound); the decomposed path sits at ~0.18 (deeply memory-bound). In practice the
fused kernel achieves 50–80% of A100's 19.5 TFLOPS peak, while the decomposed path is
bandwidth-limited to ~360 GFLOPS — a **20–40× speedup from fusion**, driven entirely by
eliminating intermediate memory traffic for the state matrix.

The 128×128 fp32 state (64 KB) requires 4 tiles of `[64, BV]` to fit in registers. FLA handles
this with unrolled tile variables — a well-understood pattern that scales across GPU generations
including Blackwell (SM100).

**Chunk-parallel kernel (prefill)** uses a multi-kernel WY decomposition pipeline:
(1) intra-chunk causal attention via dense matmul (Tensor Core friendly),
(2) inter-chunk state propagation,
(3) combined output.
Each chunk is parallelized independently; only the inter-chunk scan is sequential across $T/C$
chunks (typically $T/C = 64$ for a 4K prompt).

### 2. CPU Kernel: Simpler Than LSTM

The state matrix (128×128×4 = 64 KB per head in fp32) fits comfortably in L2 cache. The update
is element-wise — **no BLAS/GEMV dependency**, just AVX-512/NEON intrinsics:

```c
// AVX-512: tile state into 64×64 blocks (16 KB each, fits L1)
for (row = 0; row < d_k; row += 16) {
    __m512 s[d_v/16];
    load_state_tile(s, &S[row]);
    decay_tile(s, exp_g);             // S *= exp(g)
    accumulate_retrieval(retrieval, s, k[row]);  // S^T @ k
    update_tile(s, k[row], delta);    // S += k ⊗ delta
    store_state_tile(&S[row], s);
}
```

Comparison with ORT's existing `DeepCpuLstmOp`:

| | LSTM (ORT CPU) | `LinearAttention` |
|-|---------------|--------------------------|
| State size | Vector ~1–2 KB | Matrix ~64 KB |
| Core ops | 4× GEMV + sigmoid/tanh | 1 decay + 1 outer product + 1 GEMV |
| BLAS needed? | Yes (MlasGemm) | **No** — pure element-wise |
| Parallelism | batch, direction | batch, head, K-tile |
| Est. kernel size | ~50 KB C++ | ~200–400 LOC C++ |

Key advantage: linear attention heads are fully independent — embarrassingly parallel across $H$
heads. For Qwen3.5 with 32 value heads, all 32 can execute on separate cores.

### 3. NPU / Accelerator Considerations

**Recurrent decode** is a poor fit for most NPUs (Qualcomm Hexagon, Intel NPU, Apple ANE)
because of the sequential token-by-token dependency. NPUs are optimized for data-parallel
throughput, not recurrent loops.

**Chunk-parallel prefill** maps much better — the intra-chunk dense matmuls are exactly what NPU
matrix-multiply units are designed for. The inter-chunk sequential scan is short ($T/C$ steps).

**Custom silicon is ideal**: Groq (230 MB SRAM) and Cerebras WSE-3 (44 GB on-chip SRAM) can hold
the entire model's state without touching HBM. The fixed-size state never needs external memory,
unlike KV cache which may overflow SRAM for long sequences.

### 4. Runtime Dispatch: Following the Attention Op Precedent

ORT's standard `Attention` op (opset 23/24) demonstrates the ideal dispatch pattern, cascading:
`Flash Attention 2 (SM ≥ 80, fp16/bf16) → Memory-Efficient (cutlass) → Unfused math`.
The spec says nothing about flash attention — it's purely a backend optimization. `LinearAttention`
follows the same model:

```
ORT CUDA:
  ├─ Fused Triton/CUDA kernel (SM ≥ 80) — fastest (FLA-style)
  ├─ Simpler fused CUDA (SM ≥ 70) — fast
  └─ Scan decomposition fallback — correct but slow
ORT CPU:
  ├─ AVX-512 optimized kernel — fast
  └─ Reference loop — correct
TensorRT:
  ├─ IPluginV3 custom plugin — fast
  └─ Falls back to CUDA EP
```

Registration follows the standard ORT contrib op pattern —
`BuildKernelCreateInfo<LinearAttention>()` in `cuda_contrib_kernels.cc` and
`cpu_contrib_kernels.cc`, exactly as done for `Attention`, `GroupQueryAttention`, etc.
Estimated effort: 500–800 LOC for CUDA, 200–400 LOC for CPU.

### 5. Compiler Backend Lowering

TVM, IREE/MLIR, and XLA can all **express** the recurrence but **cannot auto-fuse** the full
state update through general-purpose fusion heuristics:

- **TVM**: `te.compute` generates a fused kernel for one step, but the T-token loop needs
  host-level orchestration. The WY solve for chunk-parallel is not a natural fit for polyhedral
  scheduling.
- **IREE**: The retrieval step (`S^T @ k`, a reduction) breaks the pure element-wise pattern. The
  data dependency chain (retrieve → delta → outer → update) requires custom tiling rules.
- **XLA**: `jax.lax.scan` handles the recurrence; fusion heuristics may not cover the full step.

This is exactly why a dedicated op is needed: compilers need to *recognize* the structure to apply
the right optimization. An opaque `Scan` body with 20 ops is just a bag of operations to the
compiler. A named `LinearAttention` op tells the backend what computation to optimize for.

The ONNX function body (Scan-based reference) serves dual purpose: **semantic specification** for
the op's behavior, and **universal fallback** for backends that haven't implemented native kernels.

### 6. Memory Characteristics and fp32 State Accumulation

**Fixed state vs. growing KV cache** for a Qwen3.5-like model (24 linear layers, 32 value heads,
$d_k = d_v = 128$; 8 softmax layers with GQA KV heads, fp16 storage):

Linear attention state is fixed regardless of sequence length:
- **fp16**: $24 \times 32 \times 128 \times 128 \times 2 = 24$ MB
- **fp32** (recommended for numerical stability): $24 \times 32 \times 128 \times 128 \times 4 = 48$ MB

KV cache grows linearly with sequence length:

| Sequence Length | Linear Attn State (fp32) | KV Cache ($H_{kv}=4$) | KV Cache ($H_{kv}=8$) |
|----------------|--------------------------|----------------------|----------------------|
| 1K tokens | 48 MB (fixed) | 16 MB | 32 MB |
| **3K tokens** | **48 MB** | **48 MB** | — |
| 32K tokens | 48 MB (fixed) | 512 MB | 1,024 MB |
| 128K tokens | 48 MB (fixed) | 2 GB | 4 GB |
| 1M tokens | 48 MB (fixed) | 16 GB | 32 GB |

The crossover point is typically **1.5K–3K tokens** for common GQA configurations. At 128K tokens
with $H_{kv}=4$, the KV cache is **42× larger** than the linear attention state.

**fp32 state accumulation (`S = float32`) is strongly recommended for long sequences.** The state
matrix accumulates over thousands of tokens — fp16 accumulation loses precision (or overflows
beyond 65,504) after ~1K updates. All FLA implementations universally use fp32 state, matching
LSTM's fp32 cell state convention. Callers should set the `past_state`/`present_state` dtype
(`S`) to float32 for sequences longer than ~1K tokens, while keeping activations (`T`) at
float16/bfloat16 for throughput. Runtimes handle the T↔S casting internally.

**Serving note**: For batched inference with batch size $B$, state memory scales as $B \times 48$
MB. Unlike KV cache, this is **constant** — no paging needed. Linear attention's fixed-size state
simplifies memory management for request arrival/departure (no PagedAttention-style eviction).

### 7. Fallback Strategy: The LSTM/Scan Precedent

Despite `Scan` being theoretically sufficient to express LSTM/GRU, ORT maintains native kernels
for every execution provider (`DeepCpuLstmOp` with ~50 KB of optimized C++ for CPU, custom CUDA
kernels for GPU). The performance gap between Scan execution and native kernels was too large for
practical use. **This is the exact precedent for `LinearAttention`.**

Recommended strategy:
1. Define `LinearAttention` as a first-class operator with a Scan-based **function body** for semantics and fallback
2. Runtimes implement native kernels for their EPs (porting FLA kernels for CUDA, AVX-512 for CPU)
3. Runtimes that don't recognize the op automatically expand the function body and execute via Scan — correct but 10–50× slower
4. This mirrors how many opset 18+ ONNX ops have function bodies that ORT never actually uses (because native kernels are always faster)

### Backend Implementation Summary

| Backend | Est. Effort | Native Kernel Path | Fallback |
|---------|------------|-------------------|----------|
| **ORT CUDA** | 500–800 LOC | Port FLA Triton kernels, register in `cuda_contrib_kernels.cc` | Scan expansion |
| **ORT CPU** | 200–400 LOC | AVX-512/NEON element-wise, follow `DeepCpuLstmOp` pattern | Scan expansion |
| **TensorRT** | Custom plugin | `IPluginV3` wrapping CUDA kernel (like `IRNNv2Layer` for LSTM) | CUDA EP fallback |
| **TVM** | Low | Auto-schedule single step; BYOC for external kernel lib | TE decomposition |
| **IREE/MLIR** | Medium | Pattern-match to optimized linalg + scf lowering | Generic tiled lowering |
| **OpenVINO** | Medium | `ov::Op` extension, CPU via oneDNN custom primitive | `evaluate()` reference |
| **QNN (Qualcomm)** | High | HTP graph library + HVX intrinsics | Decompose to supported ops |

The kernel engineering is well-understood, existing fused implementations are production-tested,
and the implementation cost is comparable to (or less than) what was done for LSTM/GRU across
these backends.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- **Mamba / SSD coverage**: The Mamba-2 SSD update rule uses a different state parameterization.
  Does `"gated"` cover it adequately, or should there be a `"ssd"` variant?

- **RWKV-6 full coverage**: RWKV-6 uses both token mixing and channel mixing. The `"gated"` update
  rule covers the core recurrence, but RWKV-6's full layer may need additional ops.

- **Numerical precision for state accumulation**: Should the spec allow the op to accumulate the
  state in float32 even when inputs are float16/bfloat16 (similar to how some softmax attention
  impls upcast the softmax denominator)?

- **`chunk_size` standardization**: Should `chunk_size` be a required attribute, an optional
  attribute with a default, or an implementation detail left entirely to the runtime?

## Future possibilities
[future-possibilities]: #future-possibilities

- **Mamba/SSM unification**: A future `SelectiveScan` op could cover Mamba-1 and Mamba-2's
  selective scan recurrence, which has a different structure from the matrix-state linear
  attention variants proposed here.

- **Batched-state inference**: Production serving systems batch requests with different sequence
  lengths and states. A future op variant could support variable-length batches with a mask input.

- **Block-sparse decay**: Some architectures use block-sparse or low-rank decay matrices for
  efficiency. The current packed `decay` shape `(B, T, H_{kv} \cdot d_k)` is dense; a future
  extension could accept sparse formats.

- **Hardware-specific layouts**: Future opset versions could add layout attributes (e.g., NHWC-like
  variants) for hardware backends that prefer different memory layouts for the state matrix.

## References

1. **Gated Delta Networks: Improving Mamba2 with Delta Rule** — Yang et al., ICLR 2025.
   [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
2. **The Delta Rule** (original DeltaNet) — Yang et al., 2024.
   [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
3. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** —
   Katharopoulos et al., ICML 2020. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
4. **Retentive Network: A Successor to Transformer for Large Language Models** — Sun et al., 2023.
   [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
5. **Gated Linear Attention Transformers with Hardware-Efficient Training** — Yang et al.,
   ICML 2024. [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)
6. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State
   Space Duality** (Mamba-2) — Dao & Gu, ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
7. **Jamba: A Hybrid Transformer-Mamba Language Model** — Lieber et al., 2024.
   [arXiv:2403.19887](https://arxiv.org/abs/2403.19887)
8. **RWKV: Reinventing RNNs for the Transformer Era** — Peng et al., EMNLP 2023.
   [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)
9. **flash-linear-attention** (fused Triton kernels) —
   [github.com/sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)
10. **causal-conv1d** (fused CUDA kernel) —
    [github.com/Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
11. **ONNX Attention op** (opset 23) —
    [onnx.ai/onnx/operators/onnx__Attention.html](https://onnx.ai/onnx/operators/onnx__Attention.html)
12. **FlexAttention proposal** — [github.com/onnx/onnx/issues/7494](https://github.com/onnx/onnx/issues/7494)
