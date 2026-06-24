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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_fp16() -> None:
        node = onnx.helper.make_node("Attention", inputs=["Q", "K", "V"], outputs=["Y"])

        Q = np.random.rand(2, 3, 4, 8).astype(np.float16)
        K = np.random.rand(2, 3, 6, 8).astype(np.float16)
        V = np.random.rand(2, 3, 6, 8).astype(np.float16)

        Y, _, _, _ = _compute_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_4d_fp16",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_attn_3d_mask() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 1, 4, 6).astype(np.float32)

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
            name="test_attention_4d_attn_mask_3d",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_attn_3d_mask_causal() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 1, 4, 6).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_4d_attn_mask_3d_causal",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_attn_4d_mask() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 3, 4, 6).astype(np.float32)

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
            name="test_attention_4d_attn_mask_4d",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_attn_4d_mask_causal() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 3, 4, 6).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_4d_attn_mask_4d_causal",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
        attn_mask = np.random.rand(4, 6).astype(bool)

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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_attn_mask_bool_4d() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 3, 4, 6).astype(bool)

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
            name="test_attention_4d_attn_mask_bool_4d",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_gqa_with_past_and_present_fp16() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 9, 4, 8).astype(np.float16)
        K = np.random.rand(2, 3, 6, 8).astype(np.float16)
        V = np.random.rand(2, 3, 6, 8).astype(np.float16)
        attn_mask = np.random.rand(4, 6 + past_sequence_length).astype(np.float16)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float16)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float16)

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
            name="test_attention_4d_gqa_with_past_and_present_fp16",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_diff_head_sizes_with_past_and_present_mask3D() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)
        attn_mask = np.random.rand(2, 1, 4, 6 + past_sequence_length).astype(np.float32)
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
            name="test_attention_4d_diff_heads_with_past_and_present_mask3d",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_diff_head_sizes_with_past_and_present_mask4D() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)
        attn_mask = np.random.rand(2, 3, 4, 6 + past_sequence_length).astype(np.float32)
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
            name="test_attention_4d_diff_heads_with_past_and_present_mask4d",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_with_qk_matmul_bias() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y", "", "", "qk_matmul_output"],
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
            qk_matmul_output_mode=2,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y, qk_matmul_output],
            name="test_attention_4d_with_qk_matmul_bias",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_with_qk_matmul_softcap() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y", "", "", "qk_matmul_output"],
            softcap=2.0,
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
            softcap=2.0,
            qk_matmul_output_mode=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y, qk_matmul_output],
            name="test_attention_4d_with_qk_matmul_softcap",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_with_past_and_present_qk_matmul_bias() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            qk_matmul_output_mode=2,
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
            qk_matmul_output_mode=2,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_4d_with_past_and_present_qk_matmul_bias",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_with_past_and_present_qk_matmul_bias_3d_mask() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            qk_matmul_output_mode=2,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 1, 4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            qk_matmul_output_mode=2,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_with_past_and_present_qk_matmul_bias_4d_mask() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            qk_matmul_output_mode=2,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 3, 4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            qk_matmul_output_mode=2,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_with_past_and_present_qk_matmul_bias_3d_mask_causal() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            qk_matmul_output_mode=2,
            is_causal=1,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 1, 4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            qk_matmul_output_mode=2,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_4d_with_past_and_present_qk_matmul_bias_3d_mask_causal",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_with_past_and_present_qk_matmul_bias_4d_mask_causal() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value", "qk_matmul_output"],
            qk_matmul_output_mode=2,
            is_causal=1,
        )

        past_sequence_length = 12
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)
        attn_mask = np.random.rand(2, 3, 4, 6 + past_sequence_length).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            past_key=past_key,
            past_value=past_value,
            qk_matmul_output_mode=2,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_4d_with_past_and_present_qk_matmul_bias_4d_mask_causal",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            qk_matmul_output_mode=2,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_3d_with_past_and_present_qk_matmul_bias",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            softcap=2.0,
            qk_matmul_output_mode=1,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, past_key, past_value],
            outputs=[Y, present_key, present_value, qk_matmul_output],
            name="test_attention_3d_with_past_and_present_qk_matmul_softcap",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
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
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_3d_transpose_verification() -> None:
        """Test case to verify correct 3D to 4D transpose behavior.

        This test verifies that 3D inputs are correctly reshaped and transposed
        according to the ONNX specification:
        [batch_size, seq_length, hidden_size] ->
        [batch_size, seq_length, num_heads, head_size] ->
        [batch_size, num_heads, seq_length, head_size]
        """
        q_num_heads, kv_num_heads = 3, 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
        )

        # Test inputs that will clearly demonstrate the transpose behavior
        batch_size = 1
        q_seq_length = 2
        kv_seq_length = 2
        head_size = 4
        q_hidden_size = q_num_heads * head_size  # 3 * 4 = 12
        kv_hidden_size = kv_num_heads * head_size  # 3 * 4 = 12

        # Create structured inputs to verify correct transpose behavior
        # Q has a pattern where each position in hidden dimension has a specific value
        Q = np.zeros((batch_size, q_seq_length, q_hidden_size), dtype=np.float32)
        # Fill Q with pattern: head0=[1,1,1,1], head1=[2,2,2,2], head2=[3,3,3,3]
        for head in range(q_num_heads):
            start_idx = head * head_size
            end_idx = start_idx + head_size
            Q[0, :, start_idx:end_idx] = float(head + 1)

        K = np.ones((batch_size, kv_seq_length, kv_hidden_size), dtype=np.float32) * 0.1
        V = np.ones((batch_size, kv_seq_length, kv_hidden_size), dtype=np.float32) * 0.1

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
            name="test_attention_3d_transpose_verification",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_4d_diff_heads_mask4d_padded_kv() -> None:
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"],
            outputs=["Y"],
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 10).astype(np.float32)
        attn_mask = np.random.rand(2, 3, 4, 4).astype(np.float32)
        nonpad_kv_seqlen = np.array([3, 4], dtype=np.int64)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, nonpad_kv_seqlen],
            outputs=[Y],
            name="test_attention_4d_diff_heads_mask4d_padded_kv",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_softcap_with_neginf_mask() -> None:
        """Softcap + -inf mask: verifies softcap is applied BEFORE mask/bias.

        If ordering were wrong (mask then softcap), tanh(-inf/softcap) = -1,
        so softcap * tanh(-inf/softcap) = -softcap (finite).  That leaks
        probability to masked positions.  With correct ordering (softcap then
        mask), the -inf mask values survive to softmax and yield zero weight.
        """
        np.random.seed(42)
        B, H, S_q, S_kv, D = 1, 1, 4, 6, 8

        Q = np.random.rand(B, H, S_q, D).astype(np.float32)
        K = np.random.rand(B, H, S_kv, D).astype(np.float32)
        V = np.random.rand(B, H, S_kv, D).astype(np.float32)

        # All Q positions are blocked from KV positions 4 and 5.
        attn_mask = np.zeros((S_q, S_kv), dtype=np.float32)
        attn_mask[:, 4:] = -np.inf

        softcap = 0.5

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
            softcap=softcap,
        )

        Y, _, _, _ = _compute_attention(Q, K, V, attn_mask=attn_mask, softcap=softcap)

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_4d_softcap_neginf_mask",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_softcap_with_neginf_mask_poison() -> None:
        """Softcap + -inf mask + poison values at masked KV positions.

        V has value 1000 at the masked positions (4 and 5).  With correct
        ordering the output stays in [0, 1] because the mask zeros out those
        positions.  With wrong ordering the output explodes (> 50), making
        the failure obvious even with loose tolerances.
        """
        np.random.seed(42)
        B, H, S_q, S_kv, D = 1, 1, 4, 6, 8

        Q = np.random.rand(B, H, S_q, D).astype(np.float32)
        K = np.random.rand(B, H, S_kv, D).astype(np.float32)
        V = np.random.rand(B, H, S_kv, D).astype(np.float32)

        # Block all Q positions from KV positions 4 and 5.
        attn_mask = np.zeros((S_q, S_kv), dtype=np.float32)
        attn_mask[:, 4:] = -np.inf

        # Poison: if attention leaks to masked positions, output >> 1.
        V[:, :, 4:, :] = 1000.0

        softcap = 0.5

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
            softcap=softcap,
        )

        Y, _, _, _ = _compute_attention(Q, K, V, attn_mask=attn_mask, softcap=softcap)

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_4d_softcap_neginf_mask_poison",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_4d_gqa_causal_nonpad_decode() -> None:
        """External/static-cache decode (S_q=1) with per-batch valid lengths.

        K/V are the full static cache buffer; ``nonpad_kv_seqlen`` marks how many
        leading keys are valid per batch.  With bottom-right (offset-aware) causal
        masking the single decode query attends keys ``0..nonpad[b]-1``.  Under the
        old top-left alignment it would attend only key 0, so this test fails
        pre-fix and passes post-fix.
        """
        np.random.seed(0)
        B, H_q, H_kv, L, D = 2, 4, 2, 8, 8

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "", "", "", "nonpad_kv_seqlen"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(B, H_q, 1, D).astype(np.float32)
        K = np.random.rand(B, H_kv, L, D).astype(np.float32)
        V = np.random.rand(B, H_kv, L, D).astype(np.float32)
        # Batch 0 has all 8 keys valid, batch 1 only the first 5.
        nonpad_kv_seqlen = np.array([8, 5], dtype=np.int64)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V, nonpad_kv_seqlen],
            outputs=[Y],
            name="test_attention_4d_gqa_causal_nonpad_decode",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_4d_gqa_causal_nonpad_decode_fp16() -> None:
        """fp16 variant of the external-cache decode case (locks -inf dtype handling)."""
        np.random.seed(0)
        B, H_q, H_kv, L, D = 2, 4, 2, 8, 8

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "", "", "", "nonpad_kv_seqlen"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(B, H_q, 1, D).astype(np.float16)
        K = np.random.rand(B, H_kv, L, D).astype(np.float16)
        V = np.random.rand(B, H_kv, L, D).astype(np.float16)
        nonpad_kv_seqlen = np.array([8, 5], dtype=np.int64)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V, nonpad_kv_seqlen],
            outputs=[Y],
            name="test_attention_4d_gqa_causal_nonpad_decode_fp16",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_4d_causal_nonpad_continued_prefill() -> None:
        """Continued / chunked prefill (S_q=2) into a partially-filled static cache.

        With ``nonpad_kv_seqlen = [4]`` and ``S_q = 2`` the bottom-right offset is
        ``4 - 2 = 2``: query 0 attends keys ``{0,1,2}`` and query 1 attends
        ``{0,1,2,3}``.  The old top-left alignment would mask everything past the
        diagonal (``{0}`` and ``{0,1}``), so this test fails pre-fix.
        """
        np.random.seed(1)
        B, H, L, D = 1, 2, 4, 8
        S_q = 2

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "", "", "", "nonpad_kv_seqlen"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(B, H, S_q, D).astype(np.float32)
        K = np.random.rand(B, H, L, D).astype(np.float32)
        V = np.random.rand(B, H, L, D).astype(np.float32)
        nonpad_kv_seqlen = np.array([4], dtype=np.int64)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V, nonpad_kv_seqlen],
            outputs=[Y],
            name="test_attention_4d_causal_nonpad_continued_prefill",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_4d_causal_with_past_and_present() -> None:
        """Regression guard: internal (past_key) cache + is_causal.

        This exercises the unchanged scalar bottom-right path (offset =
        past_sequence_length).  Its golden output must remain identical to the
        pre-fix behavior, proving the external-cache change does not touch the
        past_key path.
        """
        np.random.seed(2)
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
            is_causal=1,
        )

        past_sequence_length = 3
        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 4, 8).astype(np.float32)
        V = np.random.rand(2, 3, 4, 8).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, _ = _compute_attention(
            Q,
            K,
            V,
            past_key=past_key,
            past_value=past_value,
            is_causal=1,
        )

        expect(
            node,
            inputs=[Q, K, V, past_key, past_value],
            outputs=[Y, present_key, present_value],
            name="test_attention_4d_causal_with_past_and_present",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_causal_boolmask_nan_robustness() -> None:
        """Composed ``is_causal`` + boolean ``attn_mask`` NaN-robustness.

        The causal frontier (lower-triangular here, offset 0) and the boolean
        ``attn_mask`` are intersected: a key is attended only if allowed by both.
        This exercises two pre-fix NaN sources on the same forward pass:

        * **Bug-1 (allowed cells stay finite).**  Query 0 is allowed key 0 by both
          the causal frontier (``{0}``) and the mask (``True`` at key 0).  The old
          ``(1 - attn_mask) * -inf`` conversion computes ``0 * -inf = NaN`` at that
          allowed cell, poisoning the row.  The select conversion
          ``where(attn_mask, 0, -inf)`` keeps it finite.
        * **Bug-2 (fully-masked row -> 0).**  Query 1 is allowed keys ``{0, 1}`` by
          the causal frontier but the mask is ``False`` at both, so the combined
          constraint allows no key.  ``softmax`` of an all-``-inf`` row is ``NaN``;
          the fully-masked-row guard zeros it before the ``P @ V`` contraction so
          the output row is ``0``.

        4D Q/K/V is used so ``q_num_heads``/``kv_num_heads`` are omitted (passing
        them would make the function body treat the input as 3D).
        """
        np.random.seed(3)
        B, H, S, D = 1, 2, 2, 8

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(B, H, S, D).astype(np.float32)
        K = np.random.rand(B, H, S, D).astype(np.float32)
        V = np.random.rand(B, H, S, D).astype(np.float32)
        # Row 0: key 0 allowed (Bug-1 allowed cell). Row 1: no key allowed -> fully
        # masked once intersected with the causal frontier (Bug-2 empty row).
        attn_mask = np.array([[True, False], [False, False]], dtype=np.bool_)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            is_causal=1,
        )

        # Bug-1: allowed cells are finite (no NaN anywhere). Bug-2: the fully-masked
        # query row is exactly zero, not NaN.
        assert np.all(np.isfinite(Y)), "allowed cells must be finite (Bug-1)"
        assert np.array_equal(Y[:, :, 1, :], np.zeros_like(Y[:, :, 1, :])), (
            "fully-masked row must be zero (Bug-2)"
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_causal_boolmask_nan_robustness",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_23_boolmask_fullymasked_row_nan_robustness() -> None:
        """Opset-23 fully-masked boolean ``attn_mask`` row -> zero (not ``NaN``).

        This locks the opset-23 / ``old.cc`` function-body fully-masked-row guard
        against future regressions. In opset 23 the only in-contract fully-masked
        row comes from an all-``False`` boolean ``attn_mask`` row (``is_causal`` is
        not set here): every key for that query is disallowed, so ``softmax`` over an
        all-``-inf`` bias row is ``NaN``. The guard zeros that row's probabilities
        before the ``P @ V`` contraction so the output row is exactly ``0``, while
        rows with at least one allowed key are unchanged.

        4D Q/K/V is used so ``q_num_heads``/``kv_num_heads`` are omitted (passing
        them would make the function body treat the input as 3D).
        """
        np.random.seed(4)
        B, H, S, D = 1, 2, 2, 8

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y"],
        )

        Q = np.random.rand(B, H, S, D).astype(np.float32)
        K = np.random.rand(B, H, S, D).astype(np.float32)
        V = np.random.rand(B, H, S, D).astype(np.float32)
        # Row 0: no key allowed -> fully masked (Bug-2 empty row). Row 1: both keys
        # allowed -> finite, unchanged by the guard.
        attn_mask = np.array([[False, False], [True, True]], dtype=np.bool_)

        Y, _, _, _ = _compute_attention(Q, K, V, attn_mask=attn_mask)

        # Fully-masked row 0 is exactly zero (not NaN); every other cell is finite.
        assert np.all(np.isfinite(Y)), "non-masked rows must be finite"
        assert np.array_equal(Y[:, :, 0, :], np.zeros_like(Y[:, :, 0, :])), (
            "fully-masked row must be zero (Bug-2)"
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y],
            name="test_attention_23_boolmask_fullymasked_row_nan_robustness",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_4d_causal_nonpad_negative_offset_structural_empty() -> None:
        """Negative bottom-right offset: structurally-empty early query rows -> zero.

        This is the onnx-node twin of the ORT gtest
        ``Attention_Causal_NonPadKVSeqLen_StructuralEmptyRow_Zero`` /
        ``StructuralEmptyRows_Zero_CUDA``.  With ``nonpad_kv_seqlen = [2]`` and
        ``S_q = 4`` the bottom-right offset is ``2 - 4 = -2``: query row ``sq``
        attends keys ``0..(sq - 2)``, so rows 0 and 1 have an empty key set.  Their
        ``softmax`` over an all-``-inf`` bias row is ``NaN``; the fully-masked-row
        guard zeros those rows before the ``P @ V`` contraction so the output rows are
        exactly ``0``, while rows 2 and 3 (attending keys ``{0}`` and ``{0,1}``) stay finite
        and nonzero.  A ``nonpad_kv_seqlen[b] < q_sequence_length`` input is out of
        the contract's intended use, but its result is still well-defined (zeroed
        rows) rather than ``NaN``; this test pins that defined behavior.

        4D Q/K/V is used so ``q_num_heads``/``kv_num_heads`` are omitted (passing
        them would make the function body treat the input as 3D).
        """
        np.random.seed(7)
        B, H, L, D = 1, 2, 4, 8
        S_q = 4

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "", "", "", "nonpad_kv_seqlen"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(B, H, S_q, D).astype(np.float32)
        K = np.random.rand(B, H, L, D).astype(np.float32)
        V = np.random.rand(B, H, L, D).astype(np.float32)
        # offset = nonpad - S_q = 2 - 4 = -2 -> rows 0,1 structurally empty.
        nonpad_kv_seqlen = np.array([2], dtype=np.int64)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            is_causal=1,
        )

        # Structurally-empty early rows are exactly zero (not NaN); later rows finite.
        assert np.all(np.isfinite(Y)), "all output rows must be finite"
        assert np.array_equal(Y[:, :, 0, :], np.zeros_like(Y[:, :, 0, :])), (
            "structurally-empty row 0 must be zero"
        )
        assert np.array_equal(Y[:, :, 1, :], np.zeros_like(Y[:, :, 1, :])), (
            "structurally-empty row 1 must be zero"
        )
        assert np.any(Y[:, :, 2, :] != 0) and np.any(Y[:, :, 3, :] != 0), (
            "rows with a non-empty key set must be nonzero"
        )

        expect(
            node,
            inputs=[Q, K, V, nonpad_kv_seqlen],
            outputs=[Y],
            name="test_attention_4d_causal_nonpad_negative_offset_structural_empty",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_23_fullymasked_qk_matmul_output_mode3_zero() -> None:
        """Opset-23 ``qk_matmul_output_mode=3`` fully-masked row is a zero row.

        Mode ``3`` exposes the post-softmax matrix as the optional
        ``qk_matmul_output``.  For a fully-masked query row (all-``False`` boolean
        ``attn_mask`` row), the fully-masked-row guard is applied before this output
        is produced, so the mode-3 row is zeroed, consistent with the primary output
        ``Y`` row (both are ``0``).  This pins the mandated agreement between the
        guarded primary output and the mode-3 output at opset 23.

        4D Q/K/V is used so ``q_num_heads``/``kv_num_heads`` are omitted (passing
        them would make the function body treat the input as 3D).
        """
        np.random.seed(13)
        B, H, S, D = 1, 2, 2, 8

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y", "", "", "qk_matmul_output"],
            qk_matmul_output_mode=3,
        )

        Q = np.random.rand(B, H, S, D).astype(np.float32)
        K = np.random.rand(B, H, S, D).astype(np.float32)
        V = np.random.rand(B, H, S, D).astype(np.float32)
        # Row 0: no key allowed -> fully masked. Row 1: both keys allowed -> finite.
        attn_mask = np.array([[False, False], [True, True]], dtype=np.bool_)

        Y, _, _, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            qk_matmul_output_mode=3,
        )

        # Primary output row 0 and the mode-3 row 0 are both guarded to zero.
        assert np.array_equal(Y[:, :, 0, :], np.zeros_like(Y[:, :, 0, :])), (
            "fully-masked primary output row must be zero"
        )
        assert np.array_equal(
            qk_matmul_output[:, :, 0, :], np.zeros_like(qk_matmul_output[:, :, 0, :])
        ), "mode-3 output row for a fully-masked query must be zero (consistent with Y)"
        assert np.all(np.isfinite(qk_matmul_output)), (
            "all mode-3 rows are finite (the fully-masked row is guarded to 0.0)"
        )
        assert np.all(np.isfinite(Y)), (
            "all Y rows are finite (the fully-masked row is guarded to 0.0)"
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y, qk_matmul_output],
            name="test_attention_23_fullymasked_qk_matmul_output_mode3_zero",
            opset_imports=[onnx.helper.make_opsetid("", 23)],
        )

    @staticmethod
    def export_attention_24_fullymasked_qk_matmul_output_mode3_zero() -> None:
        """Opset-24 ``qk_matmul_output_mode=3`` fully-masked row is a zero row.

        The opset-24 twin of
        ``export_attention_23_fullymasked_qk_matmul_output_mode3_zero``.  Mode ``3``
        exposes the post-softmax matrix as the optional ``qk_matmul_output``.  For a
        fully-masked query row (all-``False`` boolean ``attn_mask`` row), the
        fully-masked-row guard is applied before this output is produced, so the
        mode-3 row is zeroed, consistent with the primary output ``Y`` row (both are
        ``0``).  This pins the mandated agreement between the guarded primary output
        and the mode-3 output at opset 24.

        4D Q/K/V is used so ``q_num_heads``/``kv_num_heads`` are omitted (passing
        them would make the function body treat the input as 3D).
        """
        np.random.seed(13)
        B, H, S, D = 1, 2, 2, 8

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y", "", "", "qk_matmul_output"],
            qk_matmul_output_mode=3,
        )

        Q = np.random.rand(B, H, S, D).astype(np.float32)
        K = np.random.rand(B, H, S, D).astype(np.float32)
        V = np.random.rand(B, H, S, D).astype(np.float32)
        # Row 0: no key allowed -> fully masked. Row 1: both keys allowed -> finite.
        attn_mask = np.array([[False, False], [True, True]], dtype=np.bool_)

        Y, _, _, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            qk_matmul_output_mode=3,
        )

        # Primary output row 0 and the mode-3 row 0 are both guarded to zero.
        assert np.array_equal(Y[:, :, 0, :], np.zeros_like(Y[:, :, 0, :])), (
            "fully-masked primary output row must be zero"
        )
        assert np.array_equal(
            qk_matmul_output[:, :, 0, :], np.zeros_like(qk_matmul_output[:, :, 0, :])
        ), "mode-3 output row for a fully-masked query must be zero (consistent with Y)"
        assert np.all(np.isfinite(qk_matmul_output)), (
            "all mode-3 rows are finite (the fully-masked row is guarded to 0.0)"
        )
        assert np.all(np.isfinite(Y)), (
            "all Y rows are finite (the fully-masked row is guarded to 0.0)"
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y, qk_matmul_output],
            name="test_attention_24_fullymasked_qk_matmul_output_mode3_zero",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_24_qk_matmul_output_mode3_softmax_precision() -> None:
        """Mode-3 ``qk_matmul_output`` is emitted at the output precision ``T1``.

        ``qk_matmul_output_mode=3`` exposes the post-softmax probabilities.  When
        ``softmax_precision`` differs from the operator's output type ``T1`` (here
        ``T1 = float16`` with softmax computed in ``float32``), the mode-3 output is
        cast back to ``T1`` -- matching the reference implementation, which casts the
        exposed matrix to ``Q.dtype``.  This locks both the dtype contract and the
        fully-masked-row zeroing under a non-default ``softmax_precision``.

        4D Q/K/V is used so ``q_num_heads``/``kv_num_heads`` are omitted (passing
        them would make the function body treat the input as 3D).
        """
        np.random.seed(17)
        B, H, S, D = 1, 2, 2, 8

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask"],
            outputs=["Y", "", "", "qk_matmul_output"],
            qk_matmul_output_mode=3,
            softmax_precision=int(onnx.TensorProto.FLOAT),
        )

        # T1 = float16; softmax runs in float32, so the mode-3 output is cast back to
        # float16 on emission.
        Q = np.random.rand(B, H, S, D).astype(np.float16)
        K = np.random.rand(B, H, S, D).astype(np.float16)
        V = np.random.rand(B, H, S, D).astype(np.float16)
        # Row 0: fully masked. Row 1: both keys allowed -> finite.
        attn_mask = np.array([[False, False], [True, True]], dtype=np.bool_)

        Y, _, _, qk_matmul_output = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            qk_matmul_output_mode=3,
            softmax_precision=int(onnx.TensorProto.FLOAT),
        )

        # The mode-3 output is emitted at T1 (float16), not the float32 softmax
        # precision, matching the operator's output type.
        assert qk_matmul_output.dtype == np.float16, (
            "mode-3 qk_matmul_output must be emitted at the output precision T1 (float16)"
        )
        # The fully-masked row is still guarded to zero, consistent with Y.
        assert np.array_equal(
            qk_matmul_output[:, :, 0, :], np.zeros_like(qk_matmul_output[:, :, 0, :])
        ), "mode-3 output row for a fully-masked query must be zero (consistent with Y)"
        assert np.all(np.isfinite(qk_matmul_output)), (
            "all mode-3 rows are finite (the fully-masked row is guarded to 0.0)"
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask],
            outputs=[Y, qk_matmul_output],
            name="test_attention_24_qk_matmul_output_mode3_softmax_precision",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_4d_causal_nonpad_attn_mask_composition() -> None:
        """Compose ``is_causal`` + ``nonpad_kv_seqlen`` + boolean ``attn_mask``.

        The existing nonpad tests use no ``attn_mask`` and the existing mask tests
        use no ``nonpad_kv_seqlen``; this is the first to activate all three
        constraints together on the external-cache path with ``batch > 1``.  The
        three biases are summed additively and a key is attended only if allowed by
        all three.  Crucially the inputs are designed so that **each constraint is
        independently necessary** -- removing any one changes the golden -- to avoid a
        degenerate test that a backend ignoring ``is_causal`` and/or
        ``nonpad_kv_seqlen`` could still pass:

        * **``is_causal`` binds.**  Each batch has a key that the boolean mask allows
          (``True``) but the bottom-right causal frontier disallows
          (``j > i + offset``); only ``is_causal`` masks it (batch 0 row 0 key 2,
          batch 1 row 0 key 3).
        * **``attn_mask`` binds.**  Each batch has a key the causal frontier and the
          padding bound both allow but the boolean mask sets ``False`` (batch 0 row 2
          key 1, batch 1 row 2 key 2); only the mask masks it.
        * **``nonpad_kv_seqlen`` binds.**  ``nonpad_kv_seqlen`` sets the per-batch
          causal *offset* (``offset = nonpad_kv_seqlen - q_sequence_length``), so
          dropping it collapses the frontier to top-left (``offset = 0``) and shifts
          which keys are attended.  (Under ``is_causal=1`` the causal frontier already
          subsumes the ``j < nonpad`` padding bound, so ``nonpad_kv_seqlen`` binds
          through the offset it induces rather than through a redundant padding cut.)

        The mask is chosen to leave at least one allowed key on every query row, so
        this exercises the *intersection* of the three constraints with finite outputs
        (the fully-masked-row guard is covered by
        ``test_attention_4d_causal_nonpad_negative_offset_structural_empty`` and
        ``test_attention_24_fullymasked_qk_matmul_output_mode3_zero``).

        4D Q/K/V is used so ``q_num_heads``/``kv_num_heads`` are omitted (passing
        them would make the function body treat the input as 3D).
        """
        np.random.seed(11)
        B, H, L, D = 2, 2, 6, 8
        S_q = 3

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(B, H, S_q, D).astype(np.float32)
        K = np.random.rand(B, H, L, D).astype(np.float32)
        V = np.random.rand(B, H, L, D).astype(np.float32)
        nonpad_kv_seqlen = np.array([4, 5], dtype=np.int64)  # offsets [1, 2]
        # Per-batch (B, 1, S_q, L) bool mask. Each batch is laid out so all three
        # constraints uniquely bind (see the docstring): a causal-only-masked key
        # (mask True, j > i + offset), a mask-only-masked key (mask False, causal +
        # nonpad allow it), and >=1 allowed key per row.
        attn_mask = np.array(
            [
                [
                    [
                        [True, True, True, False, False, False],
                        [True, True, True, False, False, False],
                        [True, False, True, True, False, False],
                    ]
                ],
                [
                    [
                        [True, True, True, True, False, False],
                        [True, True, True, True, False, False],
                        [True, True, False, True, True, False],
                    ]
                ],
            ],
            dtype=np.bool_,
        )

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            is_causal=1,
        )

        # The chosen mask leaves >=1 allowed key per row, so the composition stays
        # finite (no fully-masked row in this case).
        assert np.all(np.isfinite(Y)), "composed-constraint output must be finite"

        # Non-degeneracy: each constraint is independently necessary. Removing any one
        # of the three (is_causal, attn_mask, nonpad_kv_seqlen) must change the result,
        # so a backend that ignores is_causal or nonpad_kv_seqlen cannot reproduce the
        # golden by applying only the most restrictive mask.
        y_no_causal, _, _, _ = _compute_attention(
            Q, K, V, attn_mask=attn_mask, nonpad_kv_seqlen=nonpad_kv_seqlen, is_causal=0
        )
        y_no_mask, _, _, _ = _compute_attention(
            Q, K, V, nonpad_kv_seqlen=nonpad_kv_seqlen, is_causal=1
        )
        y_no_nonpad, _, _, _ = _compute_attention(
            Q, K, V, attn_mask=attn_mask, is_causal=1
        )
        assert not np.allclose(Y, y_no_causal, equal_nan=True), (
            "is_causal must bind: dropping it changes the result"
        )
        assert not np.allclose(Y, y_no_mask, equal_nan=True), (
            "attn_mask must bind: dropping it changes the result"
        )
        assert not np.allclose(Y, y_no_nonpad, equal_nan=True), (
            "nonpad_kv_seqlen must bind (via the causal offset): dropping it changes the result"
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, nonpad_kv_seqlen],
            outputs=[Y],
            name="test_attention_4d_causal_nonpad_attn_mask_composition",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_4d_causal_nonpad_batch_prefill() -> None:
        """Batch>1 continued prefill with distinct per-batch bottom-right offsets.

        The batched generalization of the ``batch == 1`` continued-prefill case: with
        ``nonpad_kv_seqlen = [4, 5, 6]`` and ``S_q = 2`` the per-batch bottom-right
        offsets are ``[2, 3, 4]`` (all ``>= 0``), so each batch realigns its causal
        frontier to its own valid-key prefix.  This pins that the per-batch offset is
        applied independently across the batch dimension.

        4D Q/K/V is used so ``q_num_heads``/``kv_num_heads`` are omitted (passing
        them would make the function body treat the input as 3D).
        """
        np.random.seed(12)
        B, H, L, D = 3, 2, 6, 8
        S_q = 2

        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "", "", "", "nonpad_kv_seqlen"],
            outputs=["Y"],
            is_causal=1,
        )

        Q = np.random.rand(B, H, S_q, D).astype(np.float32)
        K = np.random.rand(B, H, L, D).astype(np.float32)
        V = np.random.rand(B, H, L, D).astype(np.float32)
        nonpad_kv_seqlen = np.array([4, 5, 6], dtype=np.int64)  # offsets [2, 3, 4]

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            is_causal=1,
        )

        assert np.all(np.isfinite(Y)), "per-batch prefill output must be finite"

        expect(
            node,
            inputs=[Q, K, V, nonpad_kv_seqlen],
            outputs=[Y],
            name="test_attention_4d_causal_nonpad_batch_prefill",
            opset_imports=[onnx.helper.make_opsetid("", 24)],
        )

    @staticmethod
    def export_attention_local_window() -> None:
        """Sliding window attention (local_window_size=3, no explicit is_causal).

        Window implies causal: each query at position p attends only keys j
        satisfying 0 <= p - j < 3. Future positions are always masked.
        """
        local_window_size = 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            local_window_size=local_window_size,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, local_window_size=local_window_size)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_local_window",
            opset_imports=[onnx.helper.make_opsetid("", 25)],
        )

    @staticmethod
    def export_attention_local_window_default() -> None:
        """local_window_size=-1 (default/disabled) behaves identically to ver24."""
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            local_window_size=-1,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(Q, K, V, local_window_size=-1)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_local_window_default",
            opset_imports=[onnx.helper.make_opsetid("", 25)],
        )

    @staticmethod
    def export_attention_local_window_causal() -> None:
        """is_causal=1 + local_window_size=3 produces bit-identical result to
        local_window_size=3 alone (window is strict subset of causal).
        """
        local_window_size = 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            is_causal=1,
            local_window_size=local_window_size,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 6, 8).astype(np.float32)
        V = np.random.rand(2, 3, 6, 8).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q, K, V, is_causal=1, local_window_size=local_window_size
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_local_window_causal",
            opset_imports=[onnx.helper.make_opsetid("", 25)],
        )

    @staticmethod
    def export_attention_local_window_with_past() -> None:
        """Sliding window with internal KV cache (past_key/past_value)."""
        local_window_size = 3
        past_sequence_length = 8
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "", "past_key", "past_value"],
            outputs=["Y", "present_key", "present_value"],
            local_window_size=local_window_size,
        )

        Q = np.random.rand(2, 3, 4, 8).astype(np.float32)
        K = np.random.rand(2, 3, 2, 8).astype(np.float32)
        V = np.random.rand(2, 3, 2, 8).astype(np.float32)
        past_key = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)
        past_value = np.random.rand(2, 3, past_sequence_length, 8).astype(np.float32)

        Y, present_key, present_value, _ = _compute_attention(
            Q,
            K,
            V,
            past_key=past_key,
            past_value=past_value,
            local_window_size=local_window_size,
        )

        expect(
            node,
            inputs=[Q, K, V, past_key, past_value],
            outputs=[Y, present_key, present_value],
            name="test_attention_local_window_with_past",
            opset_imports=[onnx.helper.make_opsetid("", 25)],
        )

    @staticmethod
    def export_attention_local_window_ext_cache_3d_mask() -> None:
        """Sliding window + external cache + 3D attn_mask (no is_causal).

        This exercises the per-batch code path in _apply_sliding_window when
        base is 3D (batch, q, kv) because is_causal=0 means _apply_causal does
        not run, so the mask stays 3D from the attn_mask addition.
        """
        local_window_size = 3
        B, H, S_q, S_kv, D = 2, 3, 4, 8, 8
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"],
            outputs=["Y"],
            local_window_size=local_window_size,
        )

        Q = np.random.rand(B, H, S_q, D).astype(np.float32)
        K = np.random.rand(B, H, S_kv, D).astype(np.float32)
        V = np.random.rand(B, H, S_kv, D).astype(np.float32)
        # 3D mask: (batch, q, kv) — no head dimension
        attn_mask = np.random.rand(B, S_q, S_kv).astype(np.float32)
        # External cache: nonpad_kv_seqlen marks valid key count per batch
        nonpad_kv_seqlen = np.array([6, 7], dtype=np.int64)

        Y, _, _, _ = _compute_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            nonpad_kv_seqlen=nonpad_kv_seqlen,
            local_window_size=local_window_size,
        )

        expect(
            node,
            inputs=[Q, K, V, attn_mask, nonpad_kv_seqlen],
            outputs=[Y],
            name="test_attention_local_window_ext_cache_3d_mask",
            opset_imports=[onnx.helper.make_opsetid("", 25)],
        )

    @staticmethod
    def export_attention_3d_local_window() -> None:
        """Sliding window attention with 3D inputs (q_num_heads, kv_num_heads)."""
        local_window_size = 3
        node = onnx.helper.make_node(
            "Attention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            q_num_heads=3,
            kv_num_heads=3,
            local_window_size=local_window_size,
        )

        Q = np.random.rand(2, 4, 24).astype(np.float32)
        K = np.random.rand(2, 6, 24).astype(np.float32)
        V = np.random.rand(2, 6, 24).astype(np.float32)

        Y, _, _, _ = _compute_attention(
            Q, K, V, q_num_heads=3, kv_num_heads=3, local_window_size=local_window_size
        )

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_attention_3d_local_window",
            opset_imports=[onnx.helper.make_opsetid("", 25)],
        )
