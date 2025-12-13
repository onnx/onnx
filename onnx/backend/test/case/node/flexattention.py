# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.aionnx_preview.op_flex_attention import _compute_flex_attention


class FlexAttention(Base):
    @staticmethod
    def export_flex_attention_basic() -> None:
        """Test basic FlexAttention without any modifiers."""
        node = onnx.helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["output"],
            domain="ai.onnx.preview",
        )

        # Create test inputs
        batch_size, num_heads, seq_len, head_size = 2, 4, 8, 16
        Q = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float32)
        K = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float32)
        V = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float32)

        # Compute expected output using reference implementation
        output = _compute_flex_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[output],
            name="test_flex_attention_basic",
            opset_imports=[onnx.helper.make_opsetid("ai.onnx.preview", 1)],
        )

    @staticmethod
    def export_flex_attention_with_scale() -> None:
        """Test FlexAttention with custom scale factor."""
        scale_value = 0.125
        node = onnx.helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["output"],
            domain="ai.onnx.preview",
            scale=scale_value,
        )

        # Create test inputs
        batch_size, num_heads, seq_len, head_size = 2, 4, 8, 16
        Q = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float32)
        K = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float32)
        V = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float32)

        # Compute expected output using reference implementation with scale
        output = _compute_flex_attention(Q, K, V, scale=scale_value)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[output],
            name="test_flex_attention_with_scale",
            opset_imports=[onnx.helper.make_opsetid("ai.onnx.preview", 1)],
        )

    @staticmethod
    def export_flex_attention_different_seq_lengths() -> None:
        """Test FlexAttention with different query and key sequence lengths."""
        node = onnx.helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["output"],
            domain="ai.onnx.preview",
        )

        # Create test inputs with different sequence lengths
        batch_size, num_heads, seq_len_q, seq_len_k, head_size = 2, 4, 6, 10, 16
        Q = np.random.rand(batch_size, num_heads, seq_len_q, head_size).astype(np.float32)
        K = np.random.rand(batch_size, num_heads, seq_len_k, head_size).astype(np.float32)
        V = np.random.rand(batch_size, num_heads, seq_len_k, head_size).astype(np.float32)

        # Compute expected output
        output = _compute_flex_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[output],
            name="test_flex_attention_different_seq_lengths",
            opset_imports=[onnx.helper.make_opsetid("ai.onnx.preview", 1)],
        )

    @staticmethod
    def export_flex_attention_fp16() -> None:
        """Test FlexAttention with float16 precision."""
        node = onnx.helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["output"],
            domain="ai.onnx.preview",
        )

        # Create test inputs with float16
        batch_size, num_heads, seq_len, head_size = 2, 4, 8, 16
        Q = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float16)
        K = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float16)
        V = np.random.rand(batch_size, num_heads, seq_len, head_size).astype(np.float16)

        # Compute expected output
        output = _compute_flex_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[output],
            name="test_flex_attention_fp16",
            opset_imports=[onnx.helper.make_opsetid("ai.onnx.preview", 1)],
        )
