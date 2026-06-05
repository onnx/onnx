# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _unpack_3d_to_4d(x: np.ndarray, num_heads: int) -> np.ndarray:
    """Reshape (B, T, H*D) -> (B, H, T, D)."""
    b, t, hd = x.shape
    if hd % num_heads != 0:
        raise ValueError(
            f"Last dim {hd} not divisible by num_heads {num_heads} for shape {x.shape}."
        )
    d = hd // num_heads
    return x.reshape(b, t, num_heads, d).transpose(0, 2, 1, 3)


class LinearAttention(OpRun):
    def _run(
        self,
        query,
        key,
        value,
        past_state=None,
        decay=None,
        beta=None,
        chunk_size=None,  # noqa: ARG002 — tuning hint, no effect on output
        kv_num_heads=None,
        q_num_heads=None,
        scale=None,
        update_rule=None,
    ):
        # --- Step 1: defaults and validation ---
        if update_rule is None:
            update_rule = "gated_delta"
        if update_rule not in ("linear", "gated", "delta", "gated_delta"):
            raise ValueError(
                f"Unsupported update_rule '{update_rule}'. "
                "Expected one of: 'linear', 'gated', 'delta', 'gated_delta'."
            )
        if q_num_heads is None or kv_num_heads is None:
            raise ValueError("q_num_heads and kv_num_heads are required attributes.")
        if q_num_heads <= 0 or kv_num_heads <= 0 or q_num_heads % kv_num_heads != 0:
            raise ValueError(
                f"q_num_heads ({q_num_heads}) must be a positive multiple of "
                f"kv_num_heads ({kv_num_heads})."
            )

        gating = update_rule in ("gated", "gated_delta")
        delta_correction = update_rule in ("delta", "gated_delta")

        if gating and decay is None:
            raise ValueError(f"update_rule '{update_rule}' requires decay input.")
        if not gating and decay is not None:
            raise ValueError(f"update_rule '{update_rule}' forbids decay input.")
        if delta_correction and beta is None:
            raise ValueError(f"update_rule '{update_rule}' requires beta input.")
        if not delta_correction and beta is not None:
            raise ValueError(f"update_rule '{update_rule}' forbids beta input.")

        for name, arr in (("query", query), ("key", key), ("value", value)):
            if arr.ndim != 3:
                raise ValueError(
                    f"{name} must be rank 3 (B, T, H*D), got shape {arr.shape}."
                )

        # --- Step 2: unpack Q/K/V to 4D (B, H, T, D) ---
        out_dtype = query.dtype
        b, t, _ = query.shape
        d_k = query.shape[-1] // q_num_heads
        d_v = value.shape[-1] // kv_num_heads
        group_size = q_num_heads // kv_num_heads

        q4 = _unpack_3d_to_4d(query, q_num_heads).astype(np.float32)
        k4 = _unpack_3d_to_4d(key, kv_num_heads).astype(np.float32)
        v4 = _unpack_3d_to_4d(value, kv_num_heads).astype(np.float32)

        # --- Step 3: unpack decay (broadcastable to (B, H_kv, T, d_k)) ---
        if decay is not None:
            if decay.ndim != 3:
                raise ValueError(f"decay must be rank 3, got shape {decay.shape}.")
            decay_last = decay.shape[-1]
            if decay_last == kv_num_heads:
                # Per-head scalar: (B, T, H_kv) -> (B, H_kv, T, 1)
                decay4 = decay.reshape(b, t, kv_num_heads, 1).transpose(0, 2, 1, 3)
            elif decay_last == kv_num_heads * d_k:
                # Per-key-dim: (B, T, H_kv*d_k) -> (B, H_kv, T, d_k)
                decay4 = _unpack_3d_to_4d(decay, kv_num_heads)
            else:
                raise ValueError(
                    f"decay last dim {decay_last} must equal kv_num_heads "
                    f"({kv_num_heads}) or kv_num_heads*d_k ({kv_num_heads * d_k})."
                )
            decay4 = decay4.astype(np.float32)

        # --- Step 4: unpack beta (broadcastable to (B, H_kv, T, 1)) ---
        if beta is not None:
            if beta.ndim != 3:
                raise ValueError(f"beta must be rank 3, got shape {beta.shape}.")
            beta_last = beta.shape[-1]
            if beta_last not in (kv_num_heads, 1):
                raise ValueError(
                    f"beta last dim {beta_last} must be kv_num_heads "
                    f"({kv_num_heads}) or 1."
                )
            # (B, T, H_kv_or_1) -> (B, H_kv_or_1, T, 1)
            beta4 = beta.reshape(b, t, beta_last, 1).transpose(0, 2, 1, 3)
            beta4 = beta4.astype(np.float32)

        # --- Step 5: initialize state in float32 ---
        # TODO(review): The proposal allows S != T (e.g., float32 state with
        # float16/bfloat16 activations). We accumulate internally in float32
        # regardless, then cast `present_state` back to `past_state.dtype` (or
        # `query.dtype` when `past_state` is omitted, since there is no S
        # anchor in that case). A cleaner contract would propagate S
        # explicitly — possibly via a new attribute or by inferring S from a
        # zero-shape sentinel — once the spec resolves how S is signalled
        # when past_state is absent. Mirrors the same TODO in the C++
        # function-body builder in onnx/defs/nn/defs.cc.
        if past_state is not None:
            if past_state.shape != (b, kv_num_heads, d_k, d_v):
                raise ValueError(
                    f"past_state shape {past_state.shape} does not match "
                    f"({b}, {kv_num_heads}, {d_k}, {d_v})."
                )
            state_in_dtype = past_state.dtype
            state = past_state.astype(np.float32).copy()
        else:
            state_in_dtype = out_dtype
            state = np.zeros((b, kv_num_heads, d_k, d_v), dtype=np.float32)

        # --- Step 6: scale ---
        if scale is None or scale == 0.0:
            scale_val = 1.0 / np.sqrt(d_k)
        else:
            scale_val = float(scale)

        # --- Step 7+8: recurrence with GQA expansion at read time ---
        outputs = np.zeros((b, q_num_heads, t, d_v), dtype=np.float32)
        for i in range(t):
            q_t = q4[:, :, i, :]  # (B, H_q, d_k)
            k_t = k4[:, :, i, :]  # (B, H_kv, d_k)
            v_t = v4[:, :, i, :]  # (B, H_kv, d_v)

            # Decay: state *= exp(g_t)
            if gating:
                g_t = decay4[:, :, i, :]  # (B, H_kv, 1) or (B, H_kv, d_k)
                state = state * np.exp(g_t)[..., None]  # broadcast over d_v

            # Delta correction: v_t <- beta_t * (v_t - S^T @ k_t)
            if delta_correction:
                # retrieved[b, h, m] = sum_d state[b, h, d, m] * k_t[b, h, d]
                retrieved = np.einsum("bhdm,bhd->bhm", state, k_t)
                v_t = beta4[:, :, i, :] * (v_t - retrieved)

            # Write: state += k_t ⊗ v_t  (outer product over last dims)
            state = state + k_t[..., :, None] * v_t[..., None, :]

            # Read with GQA: replicate KV-head state across query heads.
            if group_size == 1:
                state_for_read = state
            else:
                # (B, H_kv, d_k, d_v) -> (B, H_q, d_k, d_v) by interleave-repeat
                state_for_read = np.repeat(state, group_size, axis=1)
            # o_t[b, h, m] = scale * sum_d q_t[b, h, d] * state_for_read[b, h, d, m]
            outputs[:, :, i, :] = scale_val * np.einsum(
                "bhd,bhdm->bhm", q_t, state_for_read
            )

        # --- Step 9: repack output (B, H_q, T, d_v) -> (B, T, H_q*d_v) ---
        output = outputs.transpose(0, 2, 1, 3).reshape(b, t, q_num_heads * d_v)
        output = output.astype(out_dtype)

        # --- Step 10: present_state in same dtype as past_state (or query) ---
        present_state = state.astype(state_in_dtype)

        return (output, present_state)
