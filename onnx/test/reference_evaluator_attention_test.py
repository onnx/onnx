# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx.reference.ops.op_attention import _compute_attention


@pytest.mark.parametrize("local_window_size", [None, -1])
def test_attention_cache_validation_without_window(local_window_size) -> None:
    q = np.zeros((1, 2, 2, 4), dtype=np.float32)
    k = np.zeros((1, 2, 3, 4), dtype=np.float32)
    v = np.zeros((1, 2, 3, 4), dtype=np.float32)
    past_key = np.zeros((1, 2, 1, 4), dtype=np.float32)
    past_value = np.zeros((1, 2, 1, 4), dtype=np.float32)

    with pytest.raises(
        ValueError, match="past_key and past_value must be provided together"
    ):
        _compute_attention(
            q, k, v, past_key=past_key, local_window_size=local_window_size
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
            local_window_size=local_window_size,
        )
