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


def _make_score_mod_bias_graph(
    bias_value: float,
    dtype: TensorProto.DataType = TensorProto.FLOAT,
) -> onnx.GraphProto:
    """Create a score_mod subgraph that adds a constant bias to the scores.

    score_mod(scores) -> scores + bias
    """
    score_in = helper.make_tensor_value_info("scores", dtype, ["B", "H", "L", "S"])
    score_out = helper.make_tensor_value_info("scores_out", dtype, ["B", "H", "L", "S"])

    bias_tensor = helper.make_tensor("bias", dtype, [], [bias_value])
    add_node = helper.make_node("Add", ["scores", "bias"], ["scores_out"])

    return helper.make_graph(
        [add_node],
        "score_mod_bias",
        [score_in],
        [score_out],
        [bias_tensor],
    )


def _make_prob_mod_scale_graph(
    scale_value: float,
    dtype: TensorProto.DataType = TensorProto.FLOAT,
) -> onnx.GraphProto:
    """Create a prob_mod subgraph that scales the probabilities.

    prob_mod(probs) -> probs * scale
    """
    prob_in = helper.make_tensor_value_info("probs", dtype, ["B", "H", "L", "S"])
    prob_out = helper.make_tensor_value_info("probs_out", dtype, ["B", "H", "L", "S"])

    scale_tensor = helper.make_tensor("scale", dtype, [], [scale_value])
    mul_node = helper.make_node("Mul", ["probs", "scale"], ["probs_out"])

    return helper.make_graph(
        [mul_node],
        "prob_mod_scale",
        [prob_in],
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
        """FlexAttention with Grouped Query Attention (GQA)."""
        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )

        B, Hq, Hkv, L, S, E, Ev = 2, 8, 2, 4, 6, 16, 16

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hkv, S, E).astype(np.float32)
        V = np.random.rand(B, Hkv, S, Ev).astype(np.float32)

        (Y,) = _compute_flex_attention(Q, K, V)

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
        score_mod_graph = _make_score_mod_bias_graph(bias_value, TensorProto.FLOAT)

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
    def export_flexattention_prob_mod() -> None:
        """FlexAttention with prob_mod subgraph (scales probabilities)."""
        scale_value = 0.5
        prob_mod_graph = _make_prob_mod_scale_graph(scale_value, TensorProto.FLOAT)

        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )
        prob_mod_attr = helper.make_attribute("prob_mod", prob_mod_graph)
        node.attribute.append(prob_mod_attr)

        B, Hq, L, E = 1, 2, 3, 4
        S, Ev = 3, 4

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        scale = 1.0 / np.sqrt(E)
        scores = np.einsum("bhle,bhse->bhls", Q, K) * scale
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        probs = probs * scale_value
        Y = np.einsum("bhls,bhsv->bhlv", probs, V).astype(np.float32)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_prob_mod",
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
