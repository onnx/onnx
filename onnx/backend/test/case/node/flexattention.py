# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx import TensorProto, helper
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.defs import AI_ONNX_PREVIEW_DOMAIN
from onnx.reference.ops.aionnx_preview.op_flex_attention import (
    _compute_flex_attention,
)


def _make_score_mod_bias_graph(bias_value: float) -> onnx.GraphProto:
    """Create a score_mod subgraph that adds a constant bias to the score.

    score_mod(score, batch, head, q_idx, k_idx) -> score + bias
    """
    # Inputs: score (float32 scalar), batch, head, q_idx, k_idx (int64 scalars)
    score_in = helper.make_tensor_value_info("score", TensorProto.FLOAT, [])
    batch_in = helper.make_tensor_value_info("batch", TensorProto.INT64, [])
    head_in = helper.make_tensor_value_info("head", TensorProto.INT64, [])
    q_idx_in = helper.make_tensor_value_info("q_idx", TensorProto.INT64, [])
    k_idx_in = helper.make_tensor_value_info("k_idx", TensorProto.INT64, [])

    # Output: score_out (float32 scalar)
    score_out = helper.make_tensor_value_info("score_out", TensorProto.FLOAT, [])

    # Constant bias
    bias_tensor = helper.make_tensor("bias", TensorProto.FLOAT, [], [bias_value])

    # Node: score_out = score + bias
    add_node = helper.make_node("Add", ["score", "bias"], ["score_out"])

    return helper.make_graph(
        [add_node],
        "score_mod_bias",
        [score_in, batch_in, head_in, q_idx_in, k_idx_in],
        [score_out],
        [bias_tensor],
    )


def _make_causal_mask_mod_graph() -> onnx.GraphProto:
    """Create a mask_mod subgraph for causal (lower triangular) masking.

    mask_mod(batch, head, q_idx, k_idx) -> q_idx >= k_idx (bool)
    """
    batch_in = helper.make_tensor_value_info("batch", TensorProto.INT64, [])
    head_in = helper.make_tensor_value_info("head", TensorProto.INT64, [])
    q_idx_in = helper.make_tensor_value_info("q_idx", TensorProto.INT64, [])
    k_idx_in = helper.make_tensor_value_info("k_idx", TensorProto.INT64, [])

    mask_out = helper.make_tensor_value_info("mask_out", TensorProto.BOOL, [])

    # Node: mask_out = q_idx >= k_idx
    ge_node = helper.make_node("GreaterOrEqual", ["q_idx", "k_idx"], ["mask_out"])

    return helper.make_graph(
        [ge_node],
        "causal_mask_mod",
        [batch_in, head_in, q_idx_in, k_idx_in],
        [mask_out],
    )


def _make_prob_mod_scale_graph(scale_value: float) -> onnx.GraphProto:
    """Create a prob_mod subgraph that scales the probability.

    prob_mod(prob, batch, head, q_idx, k_idx) -> prob * scale
    """
    prob_in = helper.make_tensor_value_info("prob", TensorProto.FLOAT, [])
    batch_in = helper.make_tensor_value_info("batch", TensorProto.INT64, [])
    head_in = helper.make_tensor_value_info("head", TensorProto.INT64, [])
    q_idx_in = helper.make_tensor_value_info("q_idx", TensorProto.INT64, [])
    k_idx_in = helper.make_tensor_value_info("k_idx", TensorProto.INT64, [])

    prob_out = helper.make_tensor_value_info("prob_out", TensorProto.FLOAT, [])

    scale_tensor = helper.make_tensor("scale", TensorProto.FLOAT, [], [scale_value])

    mul_node = helper.make_node("Mul", ["prob", "scale"], ["prob_out"])

    return helper.make_graph(
        [mul_node],
        "prob_mod_scale",
        [prob_in, batch_in, head_in, q_idx_in, k_idx_in],
        [prob_out],
        [scale_tensor],
    )


class FlexAttention(Base):
    @staticmethod
    def export_flexattention() -> None:
        """Basic FlexAttention test with default settings."""
        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )

        B, Hq, L, E = 2, 4, 8, 16
        S, Ev = 6, 16

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        (Y,) = _compute_flex_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_scaled() -> None:
        """FlexAttention with explicit scale attribute."""
        scale = 0.1
        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            scale=scale,
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )

        B, Hq, L, E = 2, 4, 8, 16
        S, Ev = 6, 16

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        (Y,) = _compute_flex_attention(Q, K, V, scale=scale)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_scaled",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_gqa() -> None:
        """FlexAttention with Grouped Query Attention (GQA) enabled."""
        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            enable_gqa=1,
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )

        B, Hq, Hkv, L, S, E, Ev = 2, 8, 2, 4, 6, 16, 16

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hkv, S, E).astype(np.float32)
        V = np.random.rand(B, Hkv, S, Ev).astype(np.float32)

        (Y,) = _compute_flex_attention(Q, K, V, enable_gqa=1)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_gqa",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_diff_head_sizes() -> None:
        """FlexAttention with different head sizes for Q/K vs V."""
        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )

        B, Hq, L, E = 2, 4, 8, 16
        S, Ev = 6, 32  # V has different head size

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        (Y,) = _compute_flex_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_diff_head_sizes",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_score_mod() -> None:
        """FlexAttention with score_mod subgraph (adds bias to scores)."""
        bias_value = 0.5
        score_mod_graph = _make_score_mod_bias_graph(bias_value)

        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )
        # Add score_mod as a graph attribute
        score_mod_attr = helper.make_attribute("score_mod", score_mod_graph)
        node.attribute.append(score_mod_attr)

        B, Hq, L, E = 1, 2, 3, 4
        S, Ev = 3, 4

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        # Reference implementation applies score_mod element-wise
        # For simplicity, compute expected output manually
        scale = 1.0 / np.sqrt(E)
        scores = np.einsum("bhle,bhse->bhls", Q, K) * scale
        scores = scores + bias_value  # score_mod: add bias
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        Y = np.einsum("bhls,bhsv->bhlv", probs, V).astype(np.float32)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_score_mod",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_mask_mod_causal() -> None:
        """FlexAttention with causal mask_mod subgraph."""
        mask_mod_graph = _make_causal_mask_mod_graph()

        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )
        mask_mod_attr = helper.make_attribute("mask_mod", mask_mod_graph)
        node.attribute.append(mask_mod_attr)

        B, Hq, L, E = 1, 2, 4, 4
        S, Ev = 4, 4

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        # Compute expected output with causal mask
        scale = 1.0 / np.sqrt(E)
        scores = np.einsum("bhle,bhse->bhls", Q, K) * scale

        # Apply causal mask: mask[q, k] = (q >= k)
        causal_mask = np.tril(np.ones((L, S), dtype=bool))
        scores = np.where(causal_mask, scores, -np.inf)

        probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = np.nan_to_num(probs)  # Handle -inf -> 0
        probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-10)
        Y = np.einsum("bhls,bhsv->bhlv", probs, V).astype(np.float32)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_mask_mod_causal",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_fp16() -> None:
        """FlexAttention with float16 inputs."""
        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )

        B, Hq, L, E = 2, 4, 8, 16
        S, Ev = 6, 16

        Q = np.random.rand(B, Hq, L, E).astype(np.float16)
        K = np.random.rand(B, Hq, S, E).astype(np.float16)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float16)

        (Y,) = _compute_flex_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_fp16",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_double() -> None:
        """FlexAttention with double precision inputs."""
        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )

        B, Hq, L, E = 2, 4, 8, 16
        S, Ev = 6, 16

        Q = np.random.rand(B, Hq, L, E).astype(np.float64)
        K = np.random.rand(B, Hq, S, E).astype(np.float64)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float64)

        (Y,) = _compute_flex_attention(Q, K, V)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_double",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )
