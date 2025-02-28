# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_attention import _compute_attention


class Attention(Base):
    @staticmethod
    def export_attention() -> None:
        node = onnx.helper.make_node("Attention", inputs=["Q", "K", "V"], outputs=["Y"])

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d",
        )

    @staticmethod
    def export_attention_gqa() -> None:
        node = onnx.helper.make_node("Attention", inputs=["Q", "K", "V"], outputs=["Y"])

        Q = np.random.rand(2, 9, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_gqa",
        )

    @staticmethod
    def export_attention_diff_head_sizes() -> None:
        node = onnx.helper.make_node("Attention", inputs=["Q", "K", "V"], outputs=["Y"])

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_diff_heads_sizes",
        )

    @staticmethod
    def export_attention_scaled() -> None:
        scale = 1e-2
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            scale=scale,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, scale=scale)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_scaled",
        )

    @staticmethod
    def export_attention_gqa_scaled() -> None:
        scale = 1e-2
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            scale=scale,
        )

        Q = np.random.rand(2, 9, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, scale=scale)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_gqa_scaled",
        )

    @staticmethod
    def export_attention_diff_head_sizes_scaled() -> None:
        scale = 1e-2
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            scale=scale,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, scale=scale)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_diff_heads_sizes_scaled",
        )

    @staticmethod
    def export_attention_causal() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, is_causal=1)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_causal",
        )

    @staticmethod
    def export_attention_gqa_causal() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(2, 9, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, is_causal=1)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_gqa_causal",
        )

    @staticmethod
    def export_attention_diff_head_sizes_causal() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_diff_heads_sizes_causal",
        )

    @staticmethod
    def export_attention_attn_mask() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_4d_attn_mask",
        )

    @staticmethod
    def export_attention_attn_mask_bool() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.bool)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_4d_attn_mask_bool",
        )

    @staticmethod
    def export_attention_gqa_attn_mask() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
        )

        Q = np.random.rand(2, 9, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_4d_gqa_attn_mask",
        )

    @staticmethod
    def export_attention_diff_head_sizes_attn_mask() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_4d_diff_heads_sizes_attn_mask",
        )

    @staticmethod
    def export_attention_with_past_and_present() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value],
            name="test_attention_4d_with_past_and_present",
        )

    @staticmethod
    def export_attention_gqa_with_past_and_present() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 9, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value],
            name="test_attention_4d_gqa_with_past_and_present",
        )

    @staticmethod
    def export_attention_diff_head_sizes_with_past_and_present() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 10).astype(np.float32)

        Y, present_key, present_value, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value],
            name="test_attention_4d_diff_heads_with_past_and_present",
        )

    @staticmethod
    def export_attention_softcap() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            softcap=2.0,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, softcap=2.0)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_softcap",
        )

    @staticmethod
    def export_attention_gqa_softcap() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            softcap=2.0,
        )

        Q = np.random.rand(2, 9, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, softcap=2.0)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_gqa_softcap",
        )

    @staticmethod
    def export_attention_diff_head_sizes_softcap() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            softcap=2.0,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            softcap=2.0,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_diff_heads_sizes_softcap",
        )

    @staticmethod
    def export_attention_with_qk_matmul() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y", "", "", "qk_matmul_output"],
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, qk_matmul_output = _compute_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y, qk_matmul_output],
            name="test_attention_4d_with_qk_matmul",
        )

    @staticmethod
    def export_attention_with_qk_matmul_bias() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y", "", "", "qk_matmul_output"],
            qk_matmul_output_mode=1,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            qk_matmul_output_mode=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y, qk_matmul_output],
            name="test_attention_4d_with_qk_matmul_bias",
        )

    @staticmethod
    def export_attention_with_qk_matmul_softcap() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y", "", "", "qk_matmul_output"],
            softcap=2.0,
            qk_matmul_output_mode=2,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            softcap=2.0,
            qk_matmul_output_mode=2,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y, qk_matmul_output],
            name="test_attention_4d_with_qk_matmul_softcap",
        )

    @staticmethod
    def export_attention_with_qk_matmul_softmax() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y", "", "", "qk_matmul_output"],
            qk_matmul_output_mode=3,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            qk_matmul_output_mode=3,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y, qk_matmul_output],
            name="test_attention_4d_with_qk_matmul_softmax",
        )

    @staticmethod
    def export_attention_with_past_and_present_qk_matmul_bias() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            qk_matmul_output_mode=1,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            qk_matmul_output_mode=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_4d_with_past_and_present_qk_matmul_bias",
        )

    @staticmethod
    def export_attention_with_past_and_present_qk_matmul() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_4d_with_past_and_present_qk_matmul",
        )

    @staticmethod
    def export_attention_3d() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d",
        )

    @staticmethod
    def export_attention_3d_gqa() -> None:
        q_num_heads, kv_num_heads = 9, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 72).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_gqa",
        )

    @staticmethod
    def export_attention_3d_diff_head_sizes() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 30).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_diff_heads_sizes",
        )

    @staticmethod
    def export_attention_3d_scaled() -> None:
        scale = 1e-2
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            scale=scale,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            scale=scale,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_scaled",
        )

    @staticmethod
    def export_attention_3d_gqa_scaled() -> None:
        scale = 1e-2
        q_num_heads, kv_num_heads = 9, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            scale=scale,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 72).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            scale=scale,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_gqa_scaled",
        )

    @staticmethod
    def export_attention_3d_diff_head_sizes_scaled() -> None:
        scale = 1e-2
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            scale=scale,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 30).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            scale=scale,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_diff_heads_sizes_scaled",
        )

    @staticmethod
    def export_attention_3d_causal() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            is_causal=1,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            is_causal=1,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_causal",
        )

    @staticmethod
    def export_attention_3d_gqa_causal() -> None:
        q_num_heads, kv_num_heads = 9, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            is_causal=1,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 72).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            is_causal=1,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_gqa_causal",
        )

    @staticmethod
    def export_attention_3d_diff_head_sizes_causal() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            is_causal=1,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 30).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            is_causal=1,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_diff_heads_sizes_causal",
        )

    @staticmethod
    def export_attention_3d_attn_mask() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_3d_attn_mask",
        )

    @staticmethod
    def export_attention_3d_gqa_attn_mask() -> None:
        q_num_heads, kv_num_heads = 9, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 72).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_3d_gqa_attn_mask",
        )

    @staticmethod
    def export_attention_3d_diff_head_sizes_attn_mask() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 30).astype(np.float32)
        attn_mask = np.random.rand(4, 6).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_3d_diff_heads_sizes_attn_mask",
        )

    @staticmethod
    def export_attention_3d_softcap() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            softcap=3.0,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            softcap=3.0,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_softcap",
        )

    @staticmethod
    def export_attention_3d_gqa_softcap() -> None:
        q_num_heads, kv_num_heads = 9, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            softcap=3.0,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 72).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            softcap=3.0,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_gqa_softcap",
        )

    @staticmethod
    def export_attention_3d_diff_head_sizes_softcap() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            softcap=3.0,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 30).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            softcap=3.0,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_diff_heads_sizes_softcap",
        )

    @staticmethod
    def export_attention_3d_with_past_and_present() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value],
            name="test_attention_3d_with_past_and_present",
        )

    @staticmethod
    def export_attention_3d_gqa_with_past_and_present() -> None:
        q_num_heads, kv_num_heads = 9, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 4, 72).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value],
            name="test_attention_3d_gqa_with_past_and_present",
        )

    @staticmethod
    def export_attention_3d_diff_head_sizes_with_past_and_present() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 30).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 10).astype(np.float32)

        Y, present_key, present_value, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value],
            name="test_attention_3d_diff_heads_with_past_and_present",
        )

    @staticmethod
    def export_attention_3d_with_past_and_present_qk_matmul() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_3d_with_past_and_present_qk_matmul",
        )

    @staticmethod
    def export_attention_3d_with_past_and_present_qk_matmul_bias() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            qk_matmul_output_mode=1,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            qk_matmul_output_mode=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_3d_with_past_and_present_qk_matmul_bias",
        )

    @staticmethod
    def export_attention_3d_with_past_and_present_qk_matmul_softcap() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            softcap=2.0,
            qk_matmul_output_mode=2,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            softcap=2.0,
            qk_matmul_output_mode=2,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_3d_with_past_and_present_qk_matmul_softcap",
        )

    @staticmethod
    def export_attention_3d_with_past_and_present_qk_matmul_softmax() -> None:
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            qk_matmul_output_mode=3,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            qk_matmul_output_mode=3,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_3d_with_past_and_present_qk_matmul_softmax",
        )
