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


def _make_score_mod_causal_mask_graph(
    dtype: TensorProto.DataType = TensorProto.FLOAT,
) -> onnx.GraphProto:
    """Create a score_mod subgraph that applies causal masking.

    For each position, masks out future tokens by setting scores to -inf
    where k_idx > q_idx. This pattern is used in Qwen-3, Gemma-3, Llama-3, etc.

    score_mod(scores) -> Where(q_idx >= k_idx, scores, -inf)
    """
    score_in = helper.make_tensor_value_info("scores", dtype, ["B", "H", "L", "S"])
    score_out = helper.make_tensor_value_info("scores_out", dtype, ["B", "H", "L", "S"])

    nodes = [
        # Extract L and S from scores shape [B, H, L, S]
        helper.make_node("Shape", ["scores"], ["scores_shape"]),
        helper.make_node("Gather", ["scores_shape", "idx_2"], ["L_dim"], axis=0),
        helper.make_node("Gather", ["scores_shape", "idx_3"], ["S_dim"], axis=0),
        # Build q_idx: range(L) reshaped to [1, 1, L, 1]
        helper.make_node("Range", ["zero", "L_dim", "one"], ["q_range"]),
        helper.make_node("Reshape", ["q_range", "q_shape"], ["q_idx"]),
        # Build k_idx: range(S) reshaped to [1, 1, 1, S]
        helper.make_node("Range", ["zero", "S_dim", "one"], ["k_range"]),
        helper.make_node("Reshape", ["k_range", "k_shape"], ["k_idx"]),
        # Causal mask: q_idx >= k_idx
        helper.make_node("GreaterOrEqual", ["q_idx", "k_idx"], ["mask"]),
        # Where(mask, scores, -inf)
        helper.make_node("Where", ["mask", "scores", "neg_inf"], ["scores_out"]),
    ]

    initializers = [
        helper.make_tensor("zero", TensorProto.INT64, [], [0]),
        helper.make_tensor("one", TensorProto.INT64, [], [1]),
        helper.make_tensor("idx_2", TensorProto.INT64, [], [2]),
        helper.make_tensor("idx_3", TensorProto.INT64, [], [3]),
        helper.make_tensor("q_shape", TensorProto.INT64, [4], [1, 1, -1, 1]),
        helper.make_tensor("k_shape", TensorProto.INT64, [4], [1, 1, 1, -1]),
        helper.make_tensor("neg_inf", dtype, [], [float("-inf")]),
    ]

    return helper.make_graph(
        nodes,
        "score_mod_causal_mask",
        [score_in],
        [score_out],
        initializers,
    )


def _make_score_mod_soft_cap_graph(
    cap_value: float,
    dtype: TensorProto.DataType = TensorProto.FLOAT,
) -> onnx.GraphProto:
    """Create a score_mod subgraph that applies soft capping.

    Used in Gemma-2 to stabilize attention scores.

    score_mod(scores) -> tanh(scores / cap) * cap
    """
    score_in = helper.make_tensor_value_info("scores", dtype, ["B", "H", "L", "S"])
    score_out = helper.make_tensor_value_info("scores_out", dtype, ["B", "H", "L", "S"])

    nodes = [
        helper.make_node("Div", ["scores", "cap"], ["scaled"]),
        helper.make_node("Tanh", ["scaled"], ["tanh_out"]),
        helper.make_node("Mul", ["tanh_out", "cap"], ["scores_out"]),
    ]

    initializers = [
        helper.make_tensor("cap", dtype, [], [cap_value]),
    ]

    return helper.make_graph(
        nodes,
        "score_mod_soft_cap",
        [score_in],
        [score_out],
        initializers,
    )


def _make_score_mod_relative_positional_graph(
    dtype: TensorProto.DataType = TensorProto.FLOAT,
) -> onnx.GraphProto:
    """Create a score_mod subgraph that adds relative positional bias.

    Adds (q_idx - k_idx) to the scores. This pattern captures the core idea
    of relative position embeddings used in various Transformer models.

    score_mod(scores) -> scores + Cast(q_idx - k_idx, dtype)
    """
    score_in = helper.make_tensor_value_info("scores", dtype, ["B", "H", "L", "S"])
    score_out = helper.make_tensor_value_info("scores_out", dtype, ["B", "H", "L", "S"])

    nodes = [
        # Extract L and S from scores shape [B, H, L, S]
        helper.make_node("Shape", ["scores"], ["scores_shape"]),
        helper.make_node("Gather", ["scores_shape", "idx_2"], ["L_dim"], axis=0),
        helper.make_node("Gather", ["scores_shape", "idx_3"], ["S_dim"], axis=0),
        # Build q_idx: range(L) reshaped to [L, 1]
        helper.make_node("Range", ["zero", "L_dim", "one"], ["q_range"]),
        helper.make_node("Reshape", ["q_range", "q_shape"], ["q_idx"]),
        # Build k_idx: range(S) reshaped to [1, S]
        helper.make_node("Range", ["zero", "S_dim", "one"], ["k_range"]),
        helper.make_node("Reshape", ["k_range", "k_shape"], ["k_idx"]),
        # Relative position: q_idx - k_idx (broadcasts to [L, S])
        helper.make_node("Sub", ["q_idx", "k_idx"], ["rel_pos"]),
        # Cast to score dtype and add to scores
        helper.make_node("Cast", ["rel_pos"], ["rel_pos_cast"], to=dtype),
        helper.make_node("Add", ["scores", "rel_pos_cast"], ["scores_out"]),
    ]

    initializers = [
        helper.make_tensor("zero", TensorProto.INT64, [], [0]),
        helper.make_tensor("one", TensorProto.INT64, [], [1]),
        helper.make_tensor("idx_2", TensorProto.INT64, [], [2]),
        helper.make_tensor("idx_3", TensorProto.INT64, [], [3]),
        helper.make_tensor("q_shape", TensorProto.INT64, [2], [-1, 1]),
        helper.make_tensor("k_shape", TensorProto.INT64, [2], [1, -1]),
    ]

    return helper.make_graph(
        nodes,
        "score_mod_relative_positional",
        [score_in],
        [score_out],
        initializers,
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

    @staticmethod
    def export_flexattention_causal_mask() -> None:
        """FlexAttention with causal masking score_mod (Qwen-3, Gemma-3, Llama-3 pattern)."""
        score_mod_graph = _make_score_mod_causal_mask_graph(TensorProto.FLOAT)

        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )
        score_mod_attr = helper.make_attribute("score_mod", score_mod_graph)
        node.attribute.append(score_mod_attr)

        B, Hq, L, E = 1, 2, 4, 8
        S, Ev = 4, 8

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        # Manually compute expected output with causal masking
        scale = 1.0 / np.sqrt(E)
        scores = np.einsum("bhle,bhse->bhls", Q, K) * scale
        # Apply causal mask: set future positions to -inf
        q_idx = np.arange(L).reshape(1, 1, L, 1)
        k_idx = np.arange(S).reshape(1, 1, 1, S)
        mask = q_idx >= k_idx
        scores = np.where(mask, scores, -np.inf)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        Y = np.einsum("bhls,bhsv->bhlv", probs, V).astype(np.float32)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_causal_mask",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_soft_cap() -> None:
        """FlexAttention with soft capping score_mod (Gemma-2 pattern)."""
        cap_value = 20.0
        score_mod_graph = _make_score_mod_soft_cap_graph(cap_value, TensorProto.FLOAT)

        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )
        score_mod_attr = helper.make_attribute("score_mod", score_mod_graph)
        node.attribute.append(score_mod_attr)

        B, Hq, L, E = 1, 2, 4, 8
        S, Ev = 4, 8

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        # Manually compute expected output with soft capping
        scale = 1.0 / np.sqrt(E)
        scores = np.einsum("bhle,bhse->bhls", Q, K) * scale
        scores = np.tanh(scores / cap_value) * cap_value
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        Y = np.einsum("bhls,bhsv->bhlv", probs, V).astype(np.float32)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_soft_cap",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )

    @staticmethod
    def export_flexattention_relative_positional() -> None:
        """FlexAttention with relative positional bias score_mod."""
        score_mod_graph = _make_score_mod_relative_positional_graph(TensorProto.FLOAT)

        node = helper.make_node(
            "FlexAttention",
            inputs=["Q", "K", "V"],
            outputs=["Y"],
            domain=AI_ONNX_PREVIEW_DOMAIN,
        )
        score_mod_attr = helper.make_attribute("score_mod", score_mod_graph)
        node.attribute.append(score_mod_attr)

        B, Hq, L, E = 1, 2, 4, 8
        S, Ev = 4, 8

        Q = np.random.rand(B, Hq, L, E).astype(np.float32)
        K = np.random.rand(B, Hq, S, E).astype(np.float32)
        V = np.random.rand(B, Hq, S, Ev).astype(np.float32)

        # Manually compute expected output with relative positional bias
        scale = 1.0 / np.sqrt(E)
        scores = np.einsum("bhle,bhse->bhls", Q, K) * scale
        q_idx = np.arange(L).reshape(-1, 1)
        k_idx = np.arange(S).reshape(1, -1)
        rel_pos = (q_idx - k_idx).astype(np.float32)
        scores = scores + rel_pos
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        Y = np.einsum("bhls,bhsv->bhlv", probs, V).astype(np.float32)

        expect(
            node,
            inputs=[Q, K, V],
            outputs=[Y],
            name="test_flexattention_relative_positional",
            opset_imports=[
                helper.make_opsetid("", 26),
                helper.make_opsetid(AI_ONNX_PREVIEW_DOMAIN, 1),
            ],
        )
