# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Union
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.backend.test.case.node.softmax import softmax


def compute_scalar_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    attn_mask: np.ndarray = None,
    past_key: np.ndarray = None, past_value: np.ndarray = None,
    scale: Union[float, None] = None,
    is_causal: bool = False,
    q_num_heads: int = 0, kv_num_heads: int = 0,
) -> np.ndarray:

    assert len(Q.shape) == len(K.shape) == len(V.shape)
    # Set input tensors (Q, K, V) to the correct shape if input shape is 3D
    # NewShapeQ (batch_size, q_num_heads, q_sequence_length, head_size)
    # NewShapeK  (batch_size, kv_num_heads, kv_sequence_length, head_size)
    # NewShapeV (value) has shape (batch_size, kv_num_heads, kv_sequence_length, v_head_size)
    batch_size = Q.shape[0]
    if len(Q.shape) == 3:
        hidden_size_q = Q.shape[2]
        hidden_size_k = K.shape[2]
        hidden_size_v = V.shape[2]
        assert q_num_heads != 0 and kv_num_heads != 0

        head_size_q = int(hidden_size_q / q_num_heads)
        new_shape_q = [batch_size, q_num_heads, Q.shape[1], head_size_q]
        Q = np.reshape(Q, new_shape_q)

        head_size_k = int(hidden_size_k / kv_num_heads)
        new_shape_k = [batch_size, kv_num_heads, K.shape[1], head_size_k]
        K = np.reshape(K, new_shape_k)

        head_size_v = int(hidden_size_v / kv_num_heads)
        new_shape_v = [batch_size, kv_num_heads, V.shape[1], head_size_v]
        V = np.reshape(V, new_shape_v)
    assert len(Q.shape) == 4 and len(K.shape) == 4 and len(V.shape) == 4

    # Calculate Scaling Factor if not provided
    if scale is None:
        q_head_size = Q.shape[3]
        scale = 1 / np.sqrt(q_head_size)

    # Update key and value cache
    if past_key is not None:
        present_key = np.concatenate((past_key, K), axis=2)
    else:
        present_key = K
    if past_value is not None:
        present_value = np.concatenate((past_value, V), axis=2)
    else:
        present_value = V

    # Create attn_bias
    q_sequence_length = Q.shape[2]
    kv_sequence_length = K.shape[2]
    attn_bias = np.zeros((q_sequence_length, kv_sequence_length), dtype=Q.dtype)
    # First case: If is_causal is provided
    # If set to true, the attention masking is a lower triangular matrix when the mask
    # is a square matrix. The attention masking has the form of the upper left causal
    # bias due to the alignment when the mask is a non-square matrix.
    if is_causal == True:
        assert attn_mask is None
        temp_mask = np.ones((q_sequence_length, kv_sequence_length), dtype=np.bool)
        temp_mask = np.tril(temp_mask, k=0)
        temp_mask = np.logical_not(temp_mask)
        attn_bias_ma = np.ma.array(attn_bias, mask=temp_mask)
        attn_bias = attn_bias_ma.filled(fill_value=float("-inf"))
    if attn_mask is not None:
        assert is_causal == False
        if attn_mask.dtype == np.bool:
            attn_mask = np.logical_not(attn_mask)
            attn_bias_ma = np.ma.array(attn_bias, mask=attn_mask)
            attn_bias = attn_bias.filled(fill_value=float("-inf"))
        else:
            attn_bias += attn_mask


    # Group Query Attention is applied if the following are satisfied
    # 1) q_num_heads != kv_num_heads
    # 2) q_num_heads % kv_num_heads == 0
    # 3) kv_num_heads == k_num_heads == v_num_heads
    q_num_heads = Q.shape[1]
    k_num_heads = K.shape[1]
    v_num_heads = K.shape[1]
    if ((q_num_heads != k_num_heads) and (q_num_heads % k_num_heads == 0) and (k_num_heads == v_num_heads)):
        seq_reps = int(q_num_heads / k_num_heads)
        reps = [1] + [seq_reps] + [1, 1]
        K = np.tile(K, reps)
        V = np.tile(V, reps)

    # The following pattern is applied
    #       Q          K          V
    #       |          |          |
    #       |      Transpose      |
    #       |          |          |
    #       ---MatMul---          |
    #             |               |
    #    scale---Mul              |
    #             |               |
    #  at_bias---Add              |
    #             |               |
    #          Softmax            |
    #             |               |
    #             -----MatMul------
    #                     |
    #                     Y
    k_transpose = np.transpose(K, (0, 1, 3, 2))
    qk = (np.matmul(Q, k_transpose) * scale) + attn_bias
    qk_softmax = softmax(qk)
    return np.matmul(qk_softmax, V), present_key, present_value


class ScalarDotProductAttention(Base):
    @staticmethod
    def export_scalar_dot_product_attention() -> None:
        node = onnx.helper.make_node(
            "ScalarDotProductAttention",
            inputs=["Q", "K", "V"],
            outputs=["output"]
        )

        q_data = np.random.rand(2, 3, 4, 8).astype(np.float32)
        k_data = np.random.rand(2, 3, 6, 8).astype(np.float32)
        v_data = np.random.rand(2, 3, 6, 8).astype(np.float32)

        expected_output, _, _ = compute_scalar_dot_product_attention(q_data, k_data, v_data)

        expect(
            node,
            inputs=[q_data, k_data, v_data],
            outputs=[expected_output],
            name="test_scalar_dot_product_attention_basic"
        )
