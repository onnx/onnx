# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def compute_rotary_embedding(
    input: np.ndarray,
    sin_cache: np.ndarray,
    cos_cache: np.ndarray,
    position_ids: np.ndarray | None,
    interleaved: int = 0,
    rotary_embedding_dim: int = 0,
    num_heads: int = 0,
) -> np.ndarray:

    # First ensure input has shape [batch_size, num_heads, seq_len, head_size]
    batch_size = input.shape[0]
    sequence_length = input.shape[1]
    if len(input.shape) == 3:
        hidden_size = input.shape[2]
        assert num_heads != 0
        head_size = int(hidden_size / num_heads)
        new_shape = [batch_size, sequence_length, num_heads, head_size]
        input = np.reshape(input, new_shape)
    assert len(input.shape) == 4
    head_size = input.shape[3]

    # Fully or partially perform rotation on input based on rotary_embedding_dim attribute
    if rotary_embedding_dim == 0:
        # If rotary_embedding_dim not provided, perform full rotation by using head_size
        rotary_embedding_dim = head_size
    x_rotate = input[:, :, :, :rotary_embedding_dim]
    x_not_rotate = input[:, :, :, rotary_embedding_dim:]
    rotary_embedding_dim_half = int(rotary_embedding_dim / 2)

    # Retrieve sin and cos caches using position ids
    if position_ids is not None:
        cos = cos_cache[position_ids]  # Shape: [batch_size, sequence_length, head_size/2]
        sin = sin_cache[position_ids]  # Shape: [batch_size, sequence_length, head_size/2]
    else:
        cos = cos_cache
        sin = sin_cache
    cos = cos[:, :, :rotary_embedding_dim_half]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    sin = sin[:, :, :rotary_embedding_dim_half]  # Shape: [batch_size, sequence_length, rotary_embedding_dim/2]
    cos = np.expand_dims(cos, axis=2) # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]
    sin = np.expand_dims(sin, axis=2) # Shape: [batch_size, sequence_length, 1, rotary_embedding_dim/2]

    # Either divide the input in halves or interleave (based on interleaved attribute)
    if interleaved:
        x1 = x_rotate[:, :, :, 0::2]
        x2 = x_rotate[:, :, :, 1::2]
    else:
        x1, x2 = np.split(x_rotate, 2, axis=-1)

    # Calculate real and imaginary values
    real = cos * x1 - sin * x2
    imag = sin * x1 + cos * x2

    # Inserted rotated embeddings back to the original input
    if interleaved:
        x_rotate[:, :, :, 0::2] = real
        x_rotate[:, :, :, 1::2] = imag
    else:
        x_rotate = np.concatenate((real, imag), axis=-1)
    output = np.concatenate((x_rotate, x_not_rotate), axis=-1)
    if len(input.shape) == 3:
        output = np.reshape(output, input.shape)
    return output


class RotaryEmbedding(Base):
    @staticmethod
    def export_rotary_embedding() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "sin_cache", "cos_cache", "position_ids"],
            outputs=["output"]
        )

        input_data = np.random.rand(2, 3, 4, 8).astype(np.float32)
        position_ids_data = np.random.rand(2, 3).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = compute_rotary_embedding(input_data, sin_cache_data, cos_cache_data, position_ids_data)

        expect(
            node,
            inputs=[input_data, sin_cache_data, cos_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding"
        )

    @staticmethod
    def export_rotary_embedding_no_position_ids() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "sin_cache", "cos_cache"],
            outputs=["output"]
        )

        input_data = np.random.rand(2, 3, 4, 8).astype(np.float32)
        sin_cache_data = np.random.rand(2, 3, 4).astype(np.float32)
        cos_cache_data = np.random.rand(2, 3, 4).astype(np.float32)

        expected_output = compute_rotary_embedding(input_data, sin_cache_data, cos_cache_data, None)

        expect(
            node,
            inputs=[input_data, sin_cache_data, cos_cache_data],
            outputs=[expected_output],
            name="test_rotary_embedding_no_position_ids"
        )

    @staticmethod
    def export_rotary_embedding_3d_input() -> None:
        num_heads = 4
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "sin_cache", "cos_cache", "position_ids"],
            outputs=["output"],
            num_heads=num_heads
        )

        input_data = np.random.rand(2, 3, 32).astype(np.float32)
        position_ids_data = np.random.rand(2, 3).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = compute_rotary_embedding(input_data, sin_cache_data, cos_cache_data, position_ids_data, num_heads=num_heads)

        expect(
            node,
            inputs=[input_data, sin_cache_data, cos_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding_3d_input"
        )

    @staticmethod
    def export_rotary_embedding_interleaved() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "sin_cache", "cos_cache", "position_ids"],
            outputs=["output"],
            interleaved=1
        )

        input_data = np.random.rand(2, 3, 4, 8).astype(np.float32)
        position_ids_data = np.random.rand(2, 3).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = compute_rotary_embedding(input_data, sin_cache_data, cos_cache_data, position_ids_data, interleaved=1)

        expect(
            node,
            inputs=[input_data, sin_cache_data, cos_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding_interleaved"
        )

    @staticmethod
    def export_rotary_embedding_with_rotary_dim() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "sin_cache", "cos_cache", "position_ids"],
            outputs=["output"],
            rotary_embedding_dim=4
        )

        input_data = np.random.rand(2, 3, 4, 8).astype(np.float32)
        position_ids_data = np.random.rand(2, 3).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = compute_rotary_embedding(input_data, sin_cache_data, cos_cache_data, position_ids_data, rotary_embedding_dim=4)

        expect(
            node,
            inputs=[input_data, sin_cache_data, cos_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding_with_rotary_dim"
        )

    @staticmethod
    def export_rotary_embedding_with_interleaved_rotary_dim() -> None:
        node = onnx.helper.make_node(
            "RotaryEmbedding",
            inputs=["input", "sin_cache", "cos_cache", "position_ids"],
            outputs=["output"],
            rotary_embedding_dim=4,
            interleaved=1,
        )

        input_data = np.random.rand(2, 3, 4, 8).astype(np.float32)
        position_ids_data = np.random.rand(2, 3).astype(np.int64)
        sin_cache_data = np.random.rand(50, 4).astype(np.float32)
        cos_cache_data = np.random.rand(50, 4).astype(np.float32)

        expected_output = compute_rotary_embedding(input_data, sin_cache_data, cos_cache_data, position_ids_data, interleaved=1, rotary_embedding_dim=4)

        expect(
            node,
            inputs=[input_data, sin_cache_data, cos_cache_data, position_ids_data],
            outputs=[expected_output],
            name="test_rotary_embedding_with_interleaved_rotary_dim"
        )
