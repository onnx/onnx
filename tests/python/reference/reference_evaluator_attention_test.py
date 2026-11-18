# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx.reference.ops._op_list import load_op
from onnx.reference.ops.op_attention import _compute_attention


def test_attention_reference_uses_versioned_implementations() -> None:
    assert load_op("", "Attention", 22).__name__ == "Attention_1"
    assert load_op("", "Attention", 23).__name__ == "Attention_23"
    assert load_op("", "Attention", 24).__name__ == "Attention_24"
    assert load_op("", "Attention", 25).__name__ == "Attention_25"


def test_attention_external_cache_rank3_mask_is_head_broadcast() -> None:
    batch_size, num_heads, q_len, kv_len, head_size = 2, 3, 2, 4, 2
    q = np.zeros((batch_size, num_heads, q_len, head_size), dtype=np.float32)
    k = np.zeros((batch_size, num_heads, kv_len, head_size), dtype=np.float32)
    v = np.arange(
        batch_size * num_heads * kv_len * head_size, dtype=np.float32
    ).reshape(batch_size, num_heads, kv_len, head_size)
    head_mask = np.zeros((num_heads, q_len, kv_len), dtype=np.float32)
    head_mask[1, :, 0] = 2.0
    nonpad_kv_seqlen = np.array([3, 4], dtype=np.int64)

    rank3_output, *_ = _compute_attention(
        q,
        k,
        v,
        attn_mask=head_mask,
        nonpad_kv_seqlen=nonpad_kv_seqlen,
        is_causal=True,
        left_window_size=2,
    )
    rank4_output, *_ = _compute_attention(
        q,
        k,
        v,
        attn_mask=head_mask[np.newaxis, ...],
        nonpad_kv_seqlen=nonpad_kv_seqlen,
        is_causal=True,
        left_window_size=2,
    )

    np.testing.assert_allclose(rank3_output, rank4_output)


def test_attention_asymmetric_bidirectional_window() -> None:
    q = np.zeros((1, 1, 5, 1), dtype=np.float32)
    k = np.zeros((1, 1, 5, 1), dtype=np.float32)
    v = np.arange(5, dtype=np.float32).reshape(1, 1, 5, 1)

    output, *_ = _compute_attention(
        q,
        k,
        v,
        left_window_size=1,
        right_window_size=2,
    )

    expected = np.array([1.0, 1.5, 2.5, 3.0, 3.5], dtype=np.float32)
    np.testing.assert_allclose(output.reshape(-1), expected)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("left_window_size", -2), ("right_window_size", -2)],
)
def test_attention_rejects_invalid_window_bounds(attribute, value) -> None:
    q = np.zeros((1, 1, 2, 1), dtype=np.float32)
    kwargs = {attribute: value}
    with pytest.raises(ValueError, match=rf"{attribute} must be -1 or nonnegative"):
        _compute_attention(q, q, q, **kwargs)


@pytest.mark.parametrize("left_window_size", [None, -1])
def test_attention_cache_validation_without_window(left_window_size) -> None:
    q = np.zeros((1, 2, 2, 4), dtype=np.float32)
    k = np.zeros((1, 2, 3, 4), dtype=np.float32)
    v = np.zeros((1, 2, 3, 4), dtype=np.float32)
    past_key = np.zeros((1, 2, 1, 4), dtype=np.float32)
    past_value = np.zeros((1, 2, 1, 4), dtype=np.float32)

    with pytest.raises(
        ValueError, match="past_key and past_value must be provided together"
    ):
        _compute_attention(
            q, k, v, past_key=past_key, left_window_size=left_window_size
        )

    with pytest.raises(
        ValueError,
        match="nonpad_kv_seqlen cannot be combined with past cache tensors",
    ):
        _compute_attention(
            q,
            k,
            v,
            past_key=past_key,
            past_value=past_value,
            nonpad_kv_seqlen=np.array([3], dtype=np.int64),
            left_window_size=left_window_size,
        )


def test_attention_cache_validation_is_preserved_for_older_opsets() -> None:
    q = np.zeros((1, 2, 2, 4), dtype=np.float32)
    attention_impl = load_op("", "Attention", 24)
    assert attention_impl._validate_attention25 is False
    with pytest.raises(
        ValueError, match="past_key and past_value must be provided together"
    ):
        _compute_attention(
            q,
            q,
            q,
            past_key=q,
            _validate_attention25=attention_impl._validate_attention25,
        )
