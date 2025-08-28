# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.reference.op_run import OpRun


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / s


def _softcap(X, softcap):
    if softcap > 0:
        Y = X / softcap
        Y = np.tanh(Y)
        return Y * softcap
    return X


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
) -> np.ndarray:
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

    # First case: If is_causal is provided
    # If set to true, the attention masking is a lower triangular matrix when the mask
    # is a square matrix. The attention masking has the form of the upper left causal
    # bias due to the alignment when the mask is a non-square matrix.
    if is_causal:
        if attn_mask is None:
            temp_mask = np.ones((q_sequence_length, kv_sequence_length), dtype=bool)
            temp_mask = np.tril(temp_mask, k=0)
            temp_mask = np.logical_not(temp_mask)
            attn_bias_ma = np.ma.array(attn_bias, mask=temp_mask)
            attn_bias = attn_bias_ma.filled(fill_value=float("-inf"))
        else:
            if attn_mask.dtype == np.bool_:
                attn_mask = (1 - attn_mask).astype(Q.dtype) * (-np.inf)
            temp_mask = np.ones((q_sequence_length, kv_sequence_length), dtype=Q.dtype)
            temp_mask = 1 - np.tril(temp_mask, k=0)
            temp_mask[temp_mask == 1] = -np.inf
            attn_bias = attn_mask + temp_mask
    elif attn_mask is not None:
        if attn_mask.dtype == np.bool_:
            attn_mask = (1 - attn_mask).astype(Q.dtype)
            attn_mask[attn_mask == 1] = -np.inf
        attn_bias = attn_bias + attn_mask

    if nonpad_kv_seqlen is not None:
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
        v_num_heads = K.shape[1]
    else:
        k_num_heads = kv_num_heads
        v_num_heads = kv_num_heads
    if (
        (q_num_heads != k_num_heads)
        and (q_num_heads % k_num_heads == 0)
        and (k_num_heads == v_num_heads)
    ):
        seq_reps = int(q_num_heads / k_num_heads)
        reps = [1, seq_reps, 1, 1]
        K = np.tile(K, reps)
        V = np.tile(V, reps)

    # The following pattern is applied
    #      Q          K          V
    #      |          |          |
    #     Q*scale    K*scale     |
    #      |          |          |
    #      |       Transpose     |
    #      |          |          |
    #      ---MatMul---          |
    #            |               |
    # at_mask---Add              |
    #            |               |
    #  softcap (if provided)     |
    #            |               |
    #         Softmax            |
    #            |               |
    #            -----MatMul------
    #                    |
    #                    Y
    k_transpose = np.transpose(K, (0, 1, 3, 2))
    qk_matmul_output = np.matmul(Q * scale, k_transpose * scale)
    qk_with_bias = qk_matmul_output + attn_bias
    if qk_matmul_output_mode == 1:
        qk_matmul_output = qk_matmul_output + attn_bias

    # Apply softcap
    if softcap is not None:
        qk_with_bias = _softcap(qk_with_bias, softcap)
        if qk_matmul_output_mode == 2:
            qk_matmul_output = qk_with_bias

    if softmax_precision is not None:
        qk_with_bias = qk_with_bias.astype(
            onnx.helper.tensor_dtype_to_np_dtype(softmax_precision)
        )
    qk_softmax = _softmax(qk_with_bias)
    if qk_matmul_output_mode == 3:
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
    ) -> np.ndarray:
        res = _compute_attention(
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
        )
        return res
