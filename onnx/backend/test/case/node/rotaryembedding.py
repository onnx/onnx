# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_rotary_embedding import rotary_embedding


class RotaryEmbedding(Base):
    @staticmethod
    def export_rotary_embedding() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "cos_cache", "sin_cache", "position_ids"],
            outputs=["output"],
        )

        input_data = np.random.rand(2, 4, 3, 8).astype(np.float32)
        position_ids_data = np.random.uniform(0, 50, (2, 3)).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = rotary_embedding(
            input_data, cos_cache_data, sin_cache_data, position_ids=position_ids_data
        )

        expect(
            node,
            inputs=[input_data, cos_cache_data, sin_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding",
        )

    @staticmethod
    def export_rotary_embedding_3d_input() -> None:
        num_heads = 4
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "cos_cache", "sin_cache", "position_ids"],
            outputs=["output"],
            num_heads=num_heads,
        )

        input_data = np.random.rand(2, 3, 32).astype(np.float32)
        position_ids_data = np.random.uniform(0, 50, (2, 3)).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            position_ids=position_ids_data,
            num_heads=num_heads,
        )

        expect(
            node,
            inputs=[input_data, cos_cache_data, sin_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding_3d_input",
        )

    @staticmethod
    def export_rotary_embedding_interleaved() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "cos_cache", "sin_cache", "position_ids"],
            outputs=["output"],
            interleaved=1,
        )

        input_data = np.random.rand(2, 4, 3, 8).astype(np.float32)
        position_ids_data = np.random.uniform(0, 50, (2, 3)).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            position_ids=position_ids_data,
            interleaved=1,
        )

        expect(
            node,
            inputs=[input_data, cos_cache_data, sin_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding_interleaved",
        )

    @staticmethod
    def export_rotary_embedding_with_rotary_dim() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "cos_cache", "sin_cache", "position_ids"],
            outputs=["output"],
            rotary_embedding_dim=4,
        )

        input_data = np.random.rand(2, 4, 3, 8).astype(np.float32)
        position_ids_data = np.random.uniform(0, 50, (2, 3)).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            position_ids=position_ids_data,
            rotary_embedding_dim=4,
        )

        expect(
            node,
            inputs=[input_data, cos_cache_data, sin_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding_with_rotary_dim",
        )

    @staticmethod
    def export_rotary_embedding_with_interleaved_rotary_dim() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "cos_cache", "sin_cache", "position_ids"],
            outputs=["output"],
            rotary_embedding_dim=4,
            interleaved=1,
        )

        input_data = np.random.rand(2, 4, 3, 8).astype(np.float32)
        position_ids_data = np.random.uniform(0, 50, (2, 3)).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            position_ids=position_ids_data,
            interleaved=1,
            rotary_embedding_dim=4,
        )

        expect(
            node,
            inputs=[input_data, cos_cache_data, sin_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding_with_interleaved_rotary_dim",
        )

    @staticmethod
    def export_rotary_embedding_no_position_ids() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "cos_cache", "sin_cache"],
            outputs=["output"],
        )

        input_data = np.random.rand(2, 4, 3, 8).astype(np.float32)
        sin_cache_data = np.random.rand(2, 3, 4).astype(np.float32)
        cos_cache_data = np.random.rand(2, 3, 4).astype(np.float32)

        expected_output = rotary_embedding(input_data, cos_cache_data, sin_cache_data)

        expect(
            node,
            inputs=[input_data, cos_cache_data, sin_cache_data],
            outputs=[expected_output],
            name="test_rotary_embedding_no_position_ids",
        )

    @staticmethod
    def export_rotary_embedding_no_position_ids_interleaved() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "cos_cache", "sin_cache"],
            outputs=["output"],
            interleaved=1,
        )

        input_data = np.random.rand(2, 4, 3, 8).astype(np.float32)
        sin_cache_data = np.random.rand(2, 3, 4).astype(np.float32)
        cos_cache_data = np.random.rand(2, 3, 4).astype(np.float32)

        expected_output = rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            interleaved=1,
        )

        expect(
            node,
            inputs=[input_data, cos_cache_data, sin_cache_data],
            outputs=[expected_output],
            name="test_rotary_embedding_no_position_ids_interleaved",
        )

    @staticmethod
    def export_rotary_embedding_no_position_ids_rotary_dim() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "cos_cache", "sin_cache"],
            outputs=["output"],
            rotary_embedding_dim=4,
        )

        input_data = np.random.rand(2, 4, 3, 8).astype(np.float32)
        sin_cache_data = np.random.rand(2, 3, 4).astype(np.float32)
        cos_cache_data = np.random.rand(2, 3, 4).astype(np.float32)

        expected_output = rotary_embedding(
            input_data,
            cos_cache_data,
            sin_cache_data,
            rotary_embedding_dim=4,
        )

        expect(
            node,
            inputs=[input_data, cos_cache_data, sin_cache_data],
            outputs=[expected_output],
            name="test_rotary_embedding_no_position_ids_rotary_dim",
        )
