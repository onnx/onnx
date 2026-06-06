<!-- Proposal: Spec gap — LinearAttention (opset 27) cannot represent Gated DeltaNet-2 -->
# Spec gap: `LinearAttention` (opset 27) cannot represent Gated DeltaNet-2's channel-wise erase gate

Follow-up to #7950 (merged) and RFC #7767.

## Summary

The newly-landed `LinearAttention` op in opset 27 covers Gated DeltaNet (v1), KDA, Mamba-2, GLA, RWKV-6 cleanly via `update_rule="gated_delta"`. However, **Gated DeltaNet-2** (NVlabs, https://github.com/NVlabs/GatedDeltaNet-2, arXiv:2605.22791) cannot be expressed through the first-class op — only through the `Scan` fallback, which is exactly the 10–50× slow path the RFC set out to avoid for hybrid LLMs.

### The GDN-2 recurrence

From the paper / README:

```
S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t−1}  +  k_t (w_t ⊙ v_t)ᵀ
```

with:
- `b_t ∈ [0,1]^{d_k}` — **channel-wise** erase gate (on the key axis)
- `w_t ∈ [0,1]^{d_v}` — **channel-wise** write gate (on the value axis)
- `D_t = Diag(α_t)` — channel-wise decay (inherited from KDA)

GDN-2's whole point is *decoupling* erase from write — they act on different axes of the state and are no longer tied to a single scalar.

### What `LinearAttention` currently does (`gated_delta`)

From `onnx/reference/ops/op_linear_attention.py` and the schema:

```
state *= exp(g_t)                    # decay, per-head scalar OR per-key-dim ✓
v_t   = beta_t * (v_t − Sᵀ k_t)      # delta correction, beta is per-head scalar
state += k_t ⊗ v_t                   # write
```

i.e. `S_t = (I − β_t k_t k_tᵀ) D_t S_{t−1} + β_t k_t v_tᵀ`, with `beta`'s last dim restricted to `{kv_num_heads, 1}` per the schema's input-validation rules.

### Per-input comparison

| GDN-2 needs | Spec today | Status |
|---|---|---|
| `D_t` channel-wise on key axis | `decay` last-dim allows `kv_num_heads * d_k` | ✓ covered |
| `w_t` channel-wise on value axis (write side only) | not in op, but the user can fold it in as a pre-`Mul`: `v ← w ⊙ v` before feeding `LinearAttention` | ✓ workable in-graph |
| `b_t` channel-wise on key axis (erase side only) | `beta` is per-head scalar; appears only as `β_t · (v_t − retrieved)` | **✗ not representable** |

The erase gate is the hard one: `b_t` lives inside the *second* `k_t` of the rank-one erase `k_t (b_t ⊙ k_t)ᵀ`. We cannot fold it into the input `k` by a pre-`Mul`, because the same `k_t` is also reused in the write term `k_t (w⊙v)ᵀ` and in the readout `q · S` — scaling `k` everywhere changes both. There is no way to express GDN-2 strictly inside the current `gated_delta` rule.

### Proposed extension (opset 28 candidate)

Two minimally-invasive options:

**(A)** Promote `beta` to allow a key-channel-wise shape (mirroring how `decay` already does):

- Allow `beta` last-dim ∈ `{1, kv_num_heads, kv_num_heads * d_k}`.
- Add a new `update_rule="gated_delta_v2"` (or relax `gated_delta`) whose function body is:

```
S_t = D_t S_{t−1} − k_t (b_t ⊙ k_tᵀ D_t S_{t−1})  +  k_t v_tᵀ
```

i.e. apply decay first, compute `retrieved = b_t ⊙ (Sᵀ k_t)` on the *key* side rather than on the output side, then subtract the rank-one update. (Note: this is **not** the same as `b_t ⊙ (v_t − retrieved)` — that scales the wrong axis.) The write gate `w_t` stays out of the op and is folded in by the user via `Mul(w, v)` in the graph.

**(B)** Add an explicit optional `erase_gate` input with shape `(B, T, kv_num_heads * d_k)` so the rules stay orthogonal. `gated_delta` (β scalar) and `gated_delta_v2` (β vector on `d_k`) coexist cleanly, and old `β_t = b_t · 1` is the strict-generalization fallback.

Either way the existing `gated_delta` behavior is preserved (backwards-compatible: GDN-2 with `b_t ≡ β_t · 1` recovers v1).

### Why this matters now

GDN-2's headline result is that it's the strongest 1.3B/100B-token model on RULER multi-key NIAH among recurrent-only and hybrid linear-attention LMs. If the architecture trend follows GDN → KDA → GDN-2, the spec gap here will repeat the situation #7950 just fixed for v1.

### References

- PR #7950 (merged): https://github.com/onnx/onnx/pull/7950
- RFC #7767: https://github.com/onnx/onnx/pull/7767
- Discussion #7689: https://github.com/onnx/onnx/issues/7689
- Gated DeltaNet-2 (NVlabs): https://github.com/NVlabs/GatedDeltaNet-2
- Paper: https://arxiv.org/abs/2605.22791

cc: reviewers/authors of #7950
