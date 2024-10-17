# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def compute_rotary_embedding(input: np.ndarray, position_ids: np.ndarray, sin_cache: np.ndarray, cos_cache: np.ndarray):
    def rotate_half(x: np.ndarray):
        x1, x2 = np.split(x, 2, axis=-1)
        return np.concatenate((-x2, x1), axis=-1)

    cos = cos_cache[position_ids]
    sin = sin_cache[position_ids]
    cos = np.expand_dims(cos, axis=1)
    sin = np.expand_dims(sin, axis=1)
    input_embed = (input * cos) + (rotate_half(input) * sin)
    return input_embed


class RotaryEmbedding(Base):
    @staticmethod
    def export_rotary_embedding() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "position_ids", "sin_cache", "cos_cache"],
            outputs=["output"]
        )

        input_data = np.random.rand(2, 3, 4).astype(np.float32)
        position_ids_data = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
        sin_cache_data = np.random.rand(3, 2).astype(np.float32)
        cos_cache_data = np.random.rand(3, 2).astype(np.float32)

        expected_output = compute_rotary_embedding(input_data, position_ids_data, sin_cache_data, cos_cache_data)

        expect(
            node,
            inputs=[input_data, position_ids_data, sin_cache_data, cos_cache_data],
            outputs=[expected_output],
            name="test_rotary_embedding"
        )

    @staticmethod
    def export_rotary_embedding_with_different_shapes() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "position_ids", "sin_cache", "cos_cache"],
            outputs=["output"]
        )

        B, SQ_LEN, dim1 = 3, 5, 6
        np.random.seed(0)
        input_data = np.random.rand(B, SQ_LEN, dim1).astype(np.float32)
        position_ids_data = np.random.randint(0, high=B, size=(B, SQ_LEN)).astype(np.int64)
        sin_cache_data = np.random.rand(SQ_LEN, dim1).astype(np.float32)
        cos_cache_data = np.random.rand(SQ_LEN, dim1).astype(np.float32)

        expected_output = compute_rotary_embedding(input_data, position_ids_data, sin_cache_data, cos_cache_data)

        expect(
            node,
            inputs=[input_data, position_ids_data, sin_cache_data, cos_cache_data],
            outputs=[expected_output],
            name="test_rotary_embedding_with_different_shapes"
        )
