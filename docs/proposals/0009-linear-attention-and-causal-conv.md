<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

- Feature Name: `linear_attention_and_causal_conv`
- Start Date: 2026-03-26
- RFC PR: [onnx/onnx#0000](https://github.com/onnx/onnx/pull/0000)
- Status: under discussion
- Authors:
  - justinchuby

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
- The state matrix is $(B, H, 128, 128)$ — **512 KB per head in fp16**. The decomposed path reads
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
                                update_rule="gated_delta", scale=0.0)

# Prefill: full sequence (same op, T > 1)
output, final_state = LinearAttention(q, k, v, initial_state, decay, beta,
                                      update_rule="gated_delta", scale=0.0)

# Packed QKV: k and v omitted, q contains concatenated QKV
output, state = LinearAttention(packed_qkv, None, None, past_state, decay, beta,
                                update_rule="gated_delta",
                                q_num_heads=16, kv_num_heads=16)
```

The operator supports both **4D unpacked** `[B, H, T, D]` and **3D packed** `[B, T, H*D]`
layouts for all inputs, consistent with the ONNX `Attention` op convention.

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
(T>1) through a single interface. Supports 3D packed and 4D unpacked input layouts.

#### Input formats

**4D inputs** (unpacked, head-explicit):

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `query` | T | $(B, H_q, T, d_k)$ | Query vectors. When `key`/`value` are absent, contains packed QKV: $(B, H_q + 2H_{kv}, T, d_k)$ |
| `key` | T (optional) | $(B, H_{kv}, T, d_k)$ | Key vectors (L2-normalized for DeltaNet variants). If absent, `query` must contain packed QKV. |
| `value` | T (optional) | $(B, H_{kv}, T, d_v)$ | Value vectors. If absent, `query` must contain packed QKV. |
| `past_state` | T (optional) | $(B, H_{kv}, d_k, d_v)$ | Recurrent state from previous step or context |
| `decay` | T (optional) | $(B, H_{kv}, T, d_k)$ | Exponential decay gate (log-space). Broadcastable: per-head scalar $(B, H_{kv}, T, 1)$ for DeltaNet/RetNet, per-key-dimension for GLA/RWKV-6. Required for `gated_*` modes. |
| `beta` | T (optional) | $(B, H_{kv}, T, 1)$ | Update rate (sigmoid output). Required for `*_delta` modes. |

**3D inputs** (packed, head-fused — requires `q_num_heads` and `kv_num_heads` attributes):

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `query` | T | $(B, T, H_q \cdot d_k)$ | Packed query vectors. When `key`/`value` are absent, contains packed QKV: $(B, T, (H_q + 2H_{kv}) \cdot d_k)$ |
| `key` | T (optional) | $(B, T, H_{kv} \cdot d_k)$ | Packed key vectors. If absent, `query` must contain packed QKV. |
| `value` | T (optional) | $(B, T, H_{kv} \cdot d_v)$ | Packed value vectors. If absent, `query` must contain packed QKV. |
| `past_state` | T (optional) | $(B, H_{kv}, d_k, d_v)$ | Recurrent state (always 4D regardless of input layout) |
| `decay` | T (optional) | $(B, T, H_{kv} \cdot d_k)$ or $(B, T, H_{kv})$ | Packed decay gates (broadcastable) |
| `beta` | T (optional) | $(B, T, H_{kv})$ or $(B, T, 1)$ | Packed update rates (broadcastable) |

The 3D layout mirrors the ONNX `Attention` op: the op internally reshapes `[B, T, H*D]` →
`[B, T, H, D]` → transposes to `[B, H, T, D]`, computes, then reshapes the output back to 3D.

#### Packed QKV

When `key` and `value` are both absent, `query` contains all three tensors concatenated along the
head dimension. The op splits them using `q_num_heads` and `kv_num_heads` (which are **required**
in this case for both 3D and 4D inputs):

- **4D packed QKV**: `query` is $(B, H_q + 2H_{kv}, T, d_k)$. Split along axis 1:
  Q = `query[:, :H_q, :, :]`, K = `query[:, H_q:H_q+H_{kv}, :, :]`, V = `query[:, H_q+H_{kv}:, :, :]`.
- **3D packed QKV**: `query` is $(B, T, (H_q + 2H_{kv}) \cdot d_k)$. Unpack to 4D first, then split on the head axis.

`key` and `value` must either both be provided or both be absent.

#### Outputs

| Name | Type | Shape (4D) | Shape (3D) | Description |
|------|------|------------|------------|-------------|
| `output` | T | $(B, H_q, T, d_v)$ | $(B, T, H_q \cdot d_v)$ | Attention output |
| `present_state` | T | $(B, H_{kv}, d_k, d_v)$ | $(B, H_{kv}, d_k, d_v)$ | Updated recurrent state (always 4D) |

#### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `q_num_heads` | int | — | Number of query heads. **Required for 3D inputs and packed QKV.** |
| `kv_num_heads` | int | — | Number of key/value heads. **Required for 3D inputs and packed QKV.** |
| `update_rule` | string | `"gated_delta"` | One of: `"linear"`, `"gated"`, `"delta"`, `"gated_delta"` |
| `scale` | float | `0.0` | Output scaling factor. When `0.0` (default), uses $1/\sqrt{d_k}$ inferred from query's last dimension. Set explicitly to override (e.g., `scale=1.0` when queries are pre-scaled by the caller). |
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

#### Reference pseudocode (`gated_delta`, 4D)

```python
def linear_attention(q, k, v, past_state, decay, beta, scale=0.0,
                     update_rule="gated_delta"):
    # q, k: (B, H, T, d_k), v: (B, H, T, d_v)
    # past_state: (B, H, d_k, d_v)
    # decay: (B, H, T, d_k) or broadcastable from (B, H, T, 1)
    # beta: (B, H, T, 1) or broadcastable
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

    output = stack(outputs, dim=2)  # (B, H, T, d_v)
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

#### Compute dtype

The op computes in the **native dtype of the inputs** (float16, bfloat16, float32). No mandatory
float32 upcast. Callers who need float32 precision for state accumulation can cast inputs before
calling the op.

---

### Operator 2: `CausalConvWithState`

Stateful causal depthwise convolution, generalized to N spatial dimensions. Used as the
preprocessing step in Gated DeltaNet and Mamba architectures.

#### Inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `input` | T | $(B, C, ...)$ | Input tensor. Spatial dims: 1D: $(L,)$; 2D: $(H, W)$; 3D: $(D, H, W)$ |
| `weight` | T | $(C, 1, k_1, ...)$ | Depthwise convolution kernel |
| `bias` | T (optional) | $(C,)$ | Per-channel bias |
| `past_state` | T (optional) | Kernel-extent prefix of spatial dims | Carry-over state from previous step |

#### Outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `output` | T | Same as `input` | Convolution output after optional activation |
| `present_state` | T | Kernel-extent prefix | Updated carry state for next call |

#### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ndim` | int | `1` | Spatial dimensionality: `1`, `2`, or `3` |
| `activation` | string | `"none"` | Optional fused activation: `"silu"`, `"swish"`, `"none"`. (`"silu"` and `"swish"` are aliases.) |

Causal padding is applied on the **left** (past-facing) side of the last spatial dimension only,
ensuring the output at position $t$ depends only on positions $\leq t$.

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

### Why 3D input support?

The ONNX `Attention` op supports both `[B, T, H*D]` (3D) and `[B, H, T, D]` (4D) layouts. Linear
attention should use the same convention: models that mix softmax and linear attention layers (like
Qwen3.5) use the same packing convention throughout, and forcing a Transpose at each layer boundary
would add unnecessary memory traffic.

### Why rank-4 decay `(B, H, T, d_k)`?

Rank-4 decay supports both per-head scalar decay (DeltaNet/RetNet: broadcast from `(B, H, T, 1)`)
and per-key-dimension decay (GLA/RWKV-6: full `(B, H, T, d_k)`). A scalar shape would not cover
GLA, which selectively retains different components of the state matrix independently.

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
- It supports both 3D and 4D inputs, handles GQA via `q_num_heads`/`kv_num_heads`, and accepts
  optional past KV and attn_mask — all design patterns carried forward in this proposal.

### Existing ONNX workaround

The current workaround (used in [onnxruntime/mobius](https://github.com/onnxruntime/mobius)) is
decomposition into ~70 primitive ONNX ops using `Scan`. This produces correct results but is
10–50× slower than fused implementations due to memory bandwidth bottlenecks.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- **Mamba / SSD coverage**: The Mamba-2 SSD update rule uses a different state parameterization.
  Does `"gated"` cover it adequately, or should there be a `"ssd"` variant?

- **RWKV-6 full coverage**: RWKV-6 uses both token mixing and channel mixing. The `"gated"` update
  rule covers the core recurrence, but RWKV-6's full layer may need additional ops.

- **Training gradients**: The chunk-parallel WY decomposition for `gated_delta` during training
  requires careful gradient computation. Should the op spec address backward pass requirements?

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
  efficiency. The current `decay` shape `(B, H, T, d_k)` is dense; a future extension could accept
  sparse formats.

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
