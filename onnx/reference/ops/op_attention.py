# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.reference.op_run import OpRun


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    # A fully-masked row is all `-inf`, so `x_max` is `-inf` and the naive
    # `exp(x - x_max)` evaluates `exp(-inf - (-inf)) = NaN`. Such a row has no
    # attendable key; by convention it softmaxes to all-zero probabilities.
    # Detect those rows on `x_max` and return 0 for them directly (instead of
    # computing NaN and overwriting it), so `_softmax` never emits a NaN/warning
    # and the all-`-inf` case is independently unit-testable.
    row_all_masked = np.isneginf(x_max)
    safe_max = np.where(row_all_masked, 0, x_max)
    tmp = np.exp(x - safe_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    s = np.where(s == 0, 1, s)  # avoid 0/0 for fully-masked rows (kept at 0)
    return tmp / s


def _softcap(X, softcap):
    if softcap > 0:
        Y = X / softcap
        Y = np.tanh(Y)
        return Y * softcap
    return X


def _apply_causal(base, offset):
    """Adds an offset-aligned (bottom-right) causal bias to `base`.

    A query at in-block index ``i`` attends key ``j`` iff ``j <= i + offset``;
    attended positions contribute ``0`` and masked positions ``-inf`` (a softmax
    then maps ``-inf`` to 0 and 0 to 1). This is the select-not-multiply form that
    mirrors the function body's ``Where(allowed, 0, -inf)`` in ``utils.cc``, so the
    reference and the ``_expanded`` graph agree bit-for-bit.

    ``offset`` is either:

    * a scalar -- the same causal frontier for every batch. Used for the internal
      cache (``offset = past_key.shape[2]``) and the no-cache case (``offset = 0``,
      i.e. ordinary top-left causal). The result keeps the 2-D ``(q, kv)`` shape of
      ``base``.
    * a 1-D per-batch array of shape ``(batch,)`` -- the external/static cache,
      where ``offset[b] = nonpad_kv_seqlen[b] - q_len``. The result is promoted to
      ``(batch, 1, q, kv)`` so the downstream padding-mask block is a no-op reshape.

    Note: ``offset`` is intentionally not clamped to ``>= 0``. A negative offset
    (the over-long-query / out-of-contract regime) fully masks the affected rows,
    which the caller's fully-masked-row guard then zeroes -- matching the spec's
    fully-masked ``Y = 0`` / mode-3 ``= 0`` behavior. Clamping would change that
    result, so it is deliberately omitted.
    """
    q_sequence_length, kv_sequence_length = base.shape[-2:]
    i_idx = np.arange(q_sequence_length).reshape(q_sequence_length, 1)  # (q, 1)
    j_idx = np.arange(kv_sequence_length).reshape(1, kv_sequence_length)  # (1, kv)
    per_batch = np.ndim(offset) > 0
    if per_batch:
        offsets = np.reshape(offset, (-1, 1, 1))  # (batch, 1, 1)
        allowed = j_idx <= (i_idx + offsets)  # (batch, q, kv)
    else:
        allowed = j_idx <= (i_idx + int(offset))  # (q, kv)
    causal = np.where(allowed, base.dtype.type(0), base.dtype.type(-np.inf))
    if per_batch:
        # Promote base to (batch, 1, q, kv) and add the per-batch causal bias.
        # For 3D (batch, q, kv), insert head axis at dim 1; for 4D, keep as-is.
        if base.ndim == 3:
            base_4d = base.reshape(base.shape[0], 1, base.shape[1], base.shape[2])
        else:
            base_4d = base.reshape((1,) * (4 - base.ndim) + base.shape)
        return base_4d + causal.reshape(
            causal.shape[0], 1, q_sequence_length, kv_sequence_length
        )
    return base + causal


def _apply_sliding_window(base, local_window_size, offset):
    """Adds a sliding-window bias to ``base``.

    Window condition: each query at absolute position ``p = offset + i`` attends
    key ``j`` iff ``0 <= p - j < local_window_size``.  This is a strict subset of
    the causal condition (``p - j >= 0``), so future positions are always masked.

    ``offset`` semantics match ``_apply_causal``: scalar for internal/no cache,
    1-D ``(batch,)`` for external cache.
    """
    q_sequence_length, kv_sequence_length = base.shape[-2:]
    i_idx = np.arange(q_sequence_length).reshape(q_sequence_length, 1)  # (q, 1)
    j_idx = np.arange(kv_sequence_length).reshape(1, kv_sequence_length)  # (1, kv)
    per_batch = np.ndim(offset) > 0
    if per_batch:
        offsets = np.reshape(offset, (-1, 1, 1))  # (batch, 1, 1)
        diff = (i_idx + offsets) - j_idx  # (batch, q, kv)
    else:
        diff = (i_idx + int(offset)) - j_idx  # (q, kv)
    allowed = (diff >= 0) & (diff < local_window_size)
    window = np.where(allowed, base.dtype.type(0), base.dtype.type(-np.inf))
    if per_batch:
        # For 3D (batch, q, kv), insert head axis at dim 1; for 4D, keep as-is.
        if base.ndim == 3:
            base_4d = base.reshape(base.shape[0], 1, base.shape[1], base.shape[2])
        else:
            base_4d = base.reshape((1,) * (4 - base.ndim) + base.shape)
        return base_4d + window.reshape(
            window.shape[0], 1, q_sequence_length, kv_sequence_length
        )
    return base + window


def _compute_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    attn_mask: np.ndarray | None = None,
    past_key: np.ndarray | None = None,
    past_value: np.ndarray | None = None,
    nonpad_kv_seqlen: np.ndarray | None = None,
    scale=None,
    is_causal=False,
    q_num_heads=None,
    kv_num_heads=None,
    softmax_precision=None,
    softcap=None,
    qk_matmul_output_mode=None,
    local_window_size=None,
) -> np.ndarray:
    if (
        local_window_size is not None
        and local_window_size != -1
        and local_window_size <= 0
    ):
        raise ValueError(
            f"local_window_size must be -1 or positive, got {local_window_size}"
        )
    assert len(Q.shape) == len(K.shape) == len(V.shape)
    # Set input tensors (Q, K, V) to the correct shape if input shape is 3D
    # NewShapeQ (batch_size, q_num_heads, q_sequence_length, head_size)
    # NewShapeK  (batch_size, kv_num_heads, kv_sequence_length, head_size)
    # NewShapeV (value) has shape (batch_size, kv_num_heads, kv_sequence_length, v_head_size)
    input_shape_len = len(Q.shape)
    batch_size = Q.shape[0]
    if len(Q.shape) == 3:
        hidden_size_q = Q.shape[2]
        hidden_size_k = K.shape[2]
        hidden_size_v = V.shape[2]
        assert q_num_heads is not None and kv_num_heads is not None

        head_size_q = int(hidden_size_q / q_num_heads)
        # First reshape to [batch_size, q_sequence_length, q_num_heads, head_size]
        intermediate_shape_q = [batch_size, Q.shape[1], q_num_heads, head_size_q]
        Q = np.reshape(Q, intermediate_shape_q)
        # Then transpose to [batch_size, q_num_heads, q_sequence_length, head_size]
        Q = np.transpose(Q, (0, 2, 1, 3))

        head_size_k = int(hidden_size_k / kv_num_heads)
        # First reshape to [batch_size, kv_sequence_length, kv_num_heads, head_size]
        intermediate_shape_k = [batch_size, K.shape[1], kv_num_heads, head_size_k]
        K = np.reshape(K, intermediate_shape_k)
        # Then transpose to [batch_size, kv_num_heads, kv_sequence_length, head_size]
        K = np.transpose(K, (0, 2, 1, 3))

        head_size_v = int(hidden_size_v / kv_num_heads)
        # First reshape to [batch_size, kv_sequence_length, kv_num_heads, head_size]
        intermediate_shape_v = [batch_size, V.shape[1], kv_num_heads, head_size_v]
        V = np.reshape(V, intermediate_shape_v)
        # Then transpose to [batch_size, kv_num_heads, kv_sequence_length, head_size]
        V = np.transpose(V, (0, 2, 1, 3))
    assert len(Q.shape) == 4 and len(K.shape) == 4 and len(V.shape) == 4

    # Calculate Scaling Factor if not provided
    if scale is None:
        q_head_size = Q.shape[3]
        scale = 1 / np.sqrt(q_head_size)
    scale = np.sqrt(scale)

    # Update key and value cache
    if past_key is not None:
        present_key = np.concatenate((past_key, K), axis=2)
    else:
        present_key = K
    if past_value is not None:
        present_value = np.concatenate((past_value, V), axis=2)
    else:
        present_value = V
    K = present_key
    V = present_value

    # Create attn_bias
    q_sequence_length = Q.shape[2]
    kv_sequence_length = K.shape[2]
    attn_bias = np.zeros((q_sequence_length, kv_sequence_length), dtype=Q.dtype)

    # The attn_mask can be less than kv_sequence_length, we need to pad it with -inf or 0
    if attn_mask is not None:
        pad_width = kv_sequence_length - attn_mask.shape[-1]
        if pad_width > 0:
            pad_shape = [(0, 0)] * (attn_mask.ndim - 1) + [(0, pad_width)]
            pad_value = False if attn_mask.dtype == np.bool_ else -np.inf
            attn_mask = np.pad(
                attn_mask, pad_shape, mode="constant", constant_values=pad_value
            )

    # First case: If is_causal is provided.
    # Causal masking is bottom-right / offset-aligned: a query at in-block index i attends
    # key j iff  j <= i + offset, where offset is the number of valid keys that precede this
    # query block. offset is derived per batch from the cache representation:
    #   * past_key present (internal cache):   offset = past_key.shape[2]   (same for all b)
    #   * nonpad_kv_seqlen present, no past_key (external/static cache):
    #                                          offset[b] = nonpad_kv_seqlen[b] - q_len
    #   * neither:                             offset = 0  (no-cache case)
    if is_causal:
        if attn_mask is not None and attn_mask.dtype == np.bool_:
            # Convert the boolean mask to an additive bias with select-not-multiply:
            # True (attend) -> 0, False (mask) -> -inf. The previous
            # (1 - attn_mask) * -inf form computes 0 * -inf = NaN at allowed cells.
            # This matches the function body's Where(attn_mask, ScalarZero, FloatNegInf)
            # in AttentionAppendFunctionCausalMask (utils.cc) so the reference and
            # _expanded agree exactly.
            attn_mask = np.where(attn_mask, Q.dtype.type(0), Q.dtype.type(-np.inf))
        base = (
            np.zeros((q_sequence_length, kv_sequence_length), dtype=Q.dtype)
            if attn_mask is None
            else attn_mask.copy()
        )
        if past_key is None and nonpad_kv_seqlen is not None:
            # External/static cache: per-batch bottom-right frontier
            #   j <= i + (nonpad_kv_seqlen[b] - q_len).
            offset = nonpad_kv_seqlen.reshape(-1) - q_sequence_length  # (batch,)
            attn_bias = _apply_causal(base, offset)
        else:
            # Internal cache (past_key) or no cache: scalar offset -- bit-identical path.
            offset = past_key.shape[2] if past_key is not None else 0
            attn_bias = _apply_causal(base, offset)
    elif attn_mask is not None:
        if attn_mask.dtype == np.bool_:
            attn_mask = (1 - attn_mask).astype(Q.dtype)
            attn_mask[attn_mask == 1] = -np.inf
        attn_bias = attn_bias + attn_mask

    # Apply sliding window mask (layered on top of causal/attn_mask)
    if local_window_size is not None and local_window_size > 0:
        if past_key is None and nonpad_kv_seqlen is not None:
            win_offset = nonpad_kv_seqlen.reshape(-1) - q_sequence_length
        elif past_key is not None:
            win_offset = past_key.shape[2]
        else:
            win_offset = 0
        attn_bias = _apply_sliding_window(attn_bias, local_window_size, win_offset)

    if nonpad_kv_seqlen is not None:
        if attn_bias.ndim == 3:
            attn_bias = attn_bias.reshape(
                attn_bias.shape[0], 1, attn_bias.shape[1], attn_bias.shape[2]
            )
        else:
            attn_bias = attn_bias.reshape(
                (1,) * (4 - attn_bias.ndim) + attn_bias.shape
            )  # broadcast to 4D
        padding_mask = np.arange(kv_sequence_length) < nonpad_kv_seqlen[:, np.newaxis]
        padding_mask = padding_mask.reshape(batch_size, 1, 1, kv_sequence_length)
        padding_mask = np.where(padding_mask, 0, -np.inf)
        attn_bias += padding_mask

    # Group Query Attention is applied if the following are satisfied
    # 1) q_num_heads != kv_num_heads
    # 2) q_num_heads % kv_num_heads == 0
    # 3) kv_num_heads == k_num_heads == v_num_heads
    if q_num_heads is None:
        q_num_heads = Q.shape[1]
    if kv_num_heads is None:
        k_num_heads = K.shape[1]
        v_num_heads = V.shape[1]
    else:
        k_num_heads = kv_num_heads
        v_num_heads = kv_num_heads
    if (
        (q_num_heads != k_num_heads)
        and (q_num_heads % k_num_heads == 0)
        and (k_num_heads == v_num_heads)
    ):
        seq_reps = q_num_heads // k_num_heads
        # Interleave-repeat each KV head: [h0, h0, h1, h1, ...]
        K = np.repeat(K, repeats=seq_reps, axis=1)
        V = np.repeat(V, repeats=seq_reps, axis=1)

    # The following pattern is applied
    #      Q          K          V
    #      |          |          |
    #     Q*scale    K*scale     |
    #      |          |          |
    #      |       Transpose     |
    #      |          |          |
    #      ---MatMul---          |
    #            |               |
    #  softcap (if provided)     |
    #            |               |
    # at_mask---Add              |
    #            |               |
    #         Softmax            |
    #            |               |
    #            -----MatMul------
    #                    |
    #                    Y
    k_transpose = np.transpose(K, (0, 1, 3, 2))
    qk_matmul_output = np.matmul(Q * scale, k_transpose * scale)

    # Apply softcap before mask/bias addition.
    # Softcap must be applied before mask so that -inf mask values remain -inf
    # (yielding zero probability in softmax). If softcap were applied after mask,
    # -inf would be mapped to -softcap (finite), leaking probability to masked positions.
    if softcap is not None:
        qk_matmul_output = _softcap(qk_matmul_output, softcap)

    qk_with_bias = qk_matmul_output + attn_bias
    if qk_matmul_output_mode == 1 and softcap is not None:
        pass  # qk_matmul_output already holds the softcapped-only value
    elif qk_matmul_output_mode == 2:
        qk_matmul_output = qk_with_bias.copy()

    if softmax_precision is not None:
        qk_with_bias = qk_with_bias.astype(
            onnx.helper.tensor_dtype_to_np_dtype(softmax_precision)
        )
    # `_softmax` is warning-free for an all-`-inf` (fully-masked) row: it returns
    # 0 instead of computing `exp(-inf - (-inf)) = NaN`. The fully-masked-row
    # semantics, however, are decided on the additive bias below (not on the
    # logits), so masking stays correct even when a masked row's raw QK score is
    # non-`-inf` (e.g. `+inf + (-inf) = NaN`).
    qk_softmax = _softmax(qk_with_bias)

    # Fully-masked-row guard: a query row whose additive bias is entirely -inf
    # (every key disallowed by the combined causal + attn_mask constraints) has no
    # attendable key. Decide this on the additive bias -- not the (possibly NaN)
    # logits -- and zero those rows with select-not-multiply (NaN * 0 = NaN) BEFORE
    # the P @ V contraction so 0 @ V = 0. The guard runs before the mode-3 capture
    # so the exposed qk_matmul_output row is also zeroed, consistent with Y, and
    # mirrors the function body's bias-based guard for primary == _expanded parity.
    row_all_masked = np.isneginf(
        np.max(attn_bias, axis=-1, keepdims=True)
    )  # (..., q, 1)
    qk_softmax = np.where(row_all_masked, 0, qk_softmax)

    if qk_matmul_output_mode == 3:
        # Mode 3 exposes the post-softmax probabilities; a fully-masked row is
        # zeroed by the guard above, consistent with the primary output Y (both
        # are 0). This matches the function body (Identity of the guarded softmax),
        # preserving primary == _expanded parity for the qk_matmul_output output.
        qk_matmul_output = qk_softmax
    qk_matmul_output = qk_matmul_output.astype(Q.dtype)

    output = np.matmul(qk_softmax, V).astype(Q.dtype)
    if input_shape_len == 3:
        output = np.transpose(output, (0, 2, 1, 3))
        output = np.reshape(output, (output.shape[0], output.shape[1], -1))
    return output, present_key, present_value, qk_matmul_output


class Attention(OpRun):
    def _run(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        attn_mask: np.ndarray | None = None,
        past_key: np.ndarray | None = None,
        past_value: np.ndarray | None = None,
        nonpad_kv_seqlen: np.ndarray | None = None,
        scale=None,
        is_causal=False,
        q_num_heads=None,
        kv_num_heads=None,
        softmax_precision=None,
        softcap=None,
        qk_matmul_output_mode=None,
        local_window_size=None,
    ) -> np.ndarray:
        return _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            scale=scale,
            is_causal=is_causal,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            softmax_precision=softmax_precision,
            softcap=softcap,
            qk_matmul_output_mode=qk_matmul_output_mode,
            local_window_size=local_window_size,
        )
