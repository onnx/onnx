# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_linear_attention import (
    LinearAttention as _RefLinearAttention,
)

_OPSET = [onnx.helper.make_opsetid("", 27)]


def _compute(
    query,
    key,
    value,
    past_state=None,
    decay=None,
    beta=None,
    *,
    q_num_heads,
    kv_num_heads,
    update_rule="gated_delta",
    scale=0.0,
    chunk_size=64,
):
    op = _RefLinearAttention.__new__(_RefLinearAttention)
    return op._run(
        query,
        key,
        value,
        past_state,
        decay,
        beta,
        chunk_size=chunk_size,
        kv_num_heads=kv_num_heads,
        q_num_heads=q_num_heads,
        scale=scale,
        update_rule=update_rule,
    )


def _l2_normalize(x: np.ndarray, num_heads: int) -> np.ndarray:
    """L2-normalize key along the per-head feature dim. Required for delta rules."""
    b, t, hd = x.shape
    d = hd // num_heads
    x4 = x.reshape(b, t, num_heads, d)
    norm = np.linalg.norm(x4, axis=-1, keepdims=True)
    x4 = x4 / np.maximum(norm, 1e-6)
    return x4.reshape(b, t, hd).astype(x.dtype)


class LinearAttention(Base):
    # ------------------------------------------------------------------
    # Update-rule axis (B=2, T=4, H_q=H_kv=4, d_k=d_v=8)
    # ------------------------------------------------------------------
    @staticmethod
    def export_linear() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value"],
            outputs=["output", "present_state"],
            update_rule="linear",
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = np.random.randn(b, t, h_kv * d_k).astype(np.float32)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
            update_rule="linear",
        )
        expect(
            node,
            inputs=[query, key, value],
            outputs=[output, present_state],
            name="test_linear_attention_linear",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_gated() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "decay"],
            outputs=["output", "present_state"],
            update_rule="gated",
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = np.random.randn(b, t, h_kv * d_k).astype(np.float32)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        # Per-key-dim decay in log-space (negative -> decay).
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1

        output, present_state = _compute(
            query,
            key,
            value,
            decay=decay,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
            update_rule="gated",
        )
        expect(
            node,
            inputs=[query, key, value, decay],
            outputs=[output, present_state],
            name="test_linear_attention_gated",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_gated_per_head_decay() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "decay"],
            outputs=["output", "present_state"],
            update_rule="gated",
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = np.random.randn(b, t, h_kv * d_k).astype(np.float32)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        # Per-head scalar decay.
        decay = -np.abs(np.random.randn(b, t, h_kv)).astype(np.float32) * 0.1

        output, present_state = _compute(
            query,
            key,
            value,
            decay=decay,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
            update_rule="gated",
        )
        expect(
            node,
            inputs=[query, key, value, decay],
            outputs=[output, present_state],
            name="test_linear_attention_gated_per_head_decay",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_delta() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "", "beta"],
            outputs=["output", "present_state"],
            update_rule="delta",
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        beta = np.random.rand(b, t, h_kv).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
            update_rule="delta",
        )
        expect(
            node,
            inputs=[query, key, value, beta],
            outputs=[output, present_state],
            name="test_linear_attention_delta",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_gated_delta() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1
        beta = np.random.rand(b, t, h_kv).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
        )
        expect(
            node,
            inputs=[query, key, value, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_gated_delta",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_gated_delta_beta_scalar() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1
        beta = np.random.rand(b, t, 1).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
        )
        expect(
            node,
            inputs=[query, key, value, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_gated_delta_beta_scalar",
            opset_imports=_OPSET,
        )

    # ------------------------------------------------------------------
    # GQA / MQA axis
    # ------------------------------------------------------------------
    @staticmethod
    def export_gated_delta_gqa() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=8,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 8, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1
        beta = np.random.rand(b, t, h_kv).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
        )
        expect(
            node,
            inputs=[query, key, value, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_gated_delta_gqa",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_gated_delta_mqa() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=8,
            kv_num_heads=1,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 8, 1, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1
        beta = np.random.rand(b, t, h_kv).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
        )
        expect(
            node,
            inputs=[query, key, value, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_gated_delta_mqa",
            opset_imports=_OPSET,
        )

    # ------------------------------------------------------------------
    # Decoding / past_state axis
    # ------------------------------------------------------------------
    @staticmethod
    def export_decode_step() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "past_state", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 1, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        past_state = np.random.randn(b, h_kv, d_k, d_v).astype(np.float32) * 0.1
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1
        beta = np.random.rand(b, t, h_kv).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            past_state=past_state,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
        )
        expect(
            node,
            inputs=[query, key, value, past_state, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_decode_step",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_prefill_with_past() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "past_state", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        past_state = np.random.randn(b, h_kv, d_k, d_v).astype(np.float32) * 0.1
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1
        beta = np.random.rand(b, t, h_kv).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            past_state=past_state,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
        )
        expect(
            node,
            inputs=[query, key, value, past_state, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_prefill_with_past",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_no_past_explicit_zeros() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "past_state", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        past_state = np.zeros((b, h_kv, d_k, d_v), dtype=np.float32)
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1
        beta = np.random.rand(b, t, h_kv).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            past_state=past_state,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
        )
        expect(
            node,
            inputs=[query, key, value, past_state, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_no_past_explicit_zeros",
            opset_imports=_OPSET,
        )

    # ------------------------------------------------------------------
    # Scale & dtype axis
    # ------------------------------------------------------------------
    @staticmethod
    def export_explicit_scale() -> None:
        scale = 0.25
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=4,
            kv_num_heads=4,
            scale=scale,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float32), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)
        decay = -np.abs(np.random.randn(b, t, h_kv * d_k)).astype(np.float32) * 0.1
        beta = np.random.rand(b, t, h_kv).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
            scale=scale,
        )
        expect(
            node,
            inputs=[query, key, value, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_explicit_scale",
            opset_imports=_OPSET,
        )

    @staticmethod
    def export_fp16() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value", "", "decay", "beta"],
            outputs=["output", "present_state"],
            q_num_heads=8,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 4, 8, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float16)
        key = _l2_normalize(np.random.randn(b, t, h_kv * d_k).astype(np.float16), h_kv)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float16)
        decay = (-np.abs(np.random.randn(b, t, h_kv * d_k)) * 0.1).astype(np.float16)
        beta = np.random.rand(b, t, h_kv).astype(np.float16)

        output, present_state = _compute(
            query,
            key,
            value,
            decay=decay,
            beta=beta,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
        )
        expect(
            node,
            inputs=[query, key, value, decay, beta],
            outputs=[output, present_state],
            name="test_linear_attention_fp16",
            opset_imports=_OPSET,
        )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------
    @staticmethod
    def export_linear_t1_no_past() -> None:
        node = onnx.helper.make_node(
            "LinearAttention",
            inputs=["query", "key", "value"],
            outputs=["output", "present_state"],
            update_rule="linear",
            q_num_heads=4,
            kv_num_heads=4,
        )
        b, t, h_q, h_kv, d_k, d_v = 2, 1, 4, 4, 8, 8
        query = np.random.randn(b, t, h_q * d_k).astype(np.float32)
        key = np.random.randn(b, t, h_kv * d_k).astype(np.float32)
        value = np.random.randn(b, t, h_kv * d_v).astype(np.float32)

        output, present_state = _compute(
            query,
            key,
            value,
            q_num_heads=h_q,
            kv_num_heads=h_kv,
            update_rule="linear",
        )
        expect(
            node,
            inputs=[query, key, value],
            outputs=[output, present_state],
            name="test_linear_attention_linear_t1_no_past",
            opset_imports=_OPSET,
        )
