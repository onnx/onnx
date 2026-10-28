# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_attention import _compute_attention


def _compute_indexed_attention(Q, K, V, indices, candidate_mask=None, scale=None):
    if any(x.ndim != 4 for x in (Q, K, V, indices)):
        raise ValueError("Q, K, V and indices must have rank 4")
    if not (Q.dtype == K.dtype == V.dtype):
        raise ValueError("Q, K and V must have identical dtypes")
    if indices.dtype not in (np.int32, np.int64):
        raise ValueError("indices must have int32 or int64 dtype")
    batch, heads, queries, depth = Q.shape
    key_heads, history = K.shape[1:3]
    value_depth = V.shape[3]
    if min(batch, heads, queries, depth, key_heads, value_depth) <= 0:
        raise ValueError(
            "batch, head counts, query length and head sizes must be positive"
        )
    if (
        K.shape[0] != batch
        or V.shape[:3] != K.shape[:3]
        or K.shape[3] != depth
        or heads % key_heads
        or indices.shape[0] != batch
        or indices.shape[1] not in (1, heads)
        or indices.shape[2] != queries
    ):
        raise ValueError("incompatible IndexedAttention input shapes or head counts")
    if scale is not None and not scale >= 0:
        raise ValueError("scale must be nonnegative")
    count = indices.shape[3]
    indices = np.broadcast_to(indices.astype(np.int64), (batch, heads, queries, count))
    allowed = (indices >= 0) & (indices < history)
    if candidate_mask is not None:
        if candidate_mask.dtype != np.bool_:
            raise ValueError("candidate_mask must have boolean dtype")
        allowed = allowed & np.broadcast_to(candidate_mask, indices.shape)
    allowed = np.pad(allowed, ((0, 0), (0, 0), (0, 0), (0, 1)))
    indices = np.pad(indices, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=-1)
    safe_indices = np.where(allowed, indices + 1, 0)
    batch_indices = np.arange(batch)[:, None, None, None]
    head_indices = (np.arange(heads) // (heads // key_heads))[None, :, None, None]
    padded_k = np.pad(K, ((0, 0), (0, 0), (1, 0), (0, 0)))
    padded_v = np.pad(V, ((0, 0), (0, 0), (1, 0), (0, 0)))
    selected_k = padded_k[batch_indices, head_indices, safe_indices]
    selected_v = padded_v[batch_indices, head_indices, safe_indices]
    accumulation_dtype = np.float64 if Q.dtype == np.float64 else np.float32
    rows = batch * heads * queries
    output, *_ = _compute_attention(
        Q.reshape(rows, 1, 1, depth).astype(accumulation_dtype),
        selected_k.reshape(rows, 1, count + 1, depth).astype(accumulation_dtype),
        selected_v.reshape(rows, 1, count + 1, value_depth).astype(accumulation_dtype),
        attn_mask=allowed.reshape(rows, 1, 1, count + 1),
        scale=scale,
    )
    return (output.reshape(batch, heads, queries, value_depth).astype(Q.dtype),)


class IndexedAttention_29(OpRun):
    def _run(self, Q, K, V, indices, candidate_mask=None, scale=None):
        return _compute_indexed_attention(Q, K, V, indices, candidate_mask, scale)
