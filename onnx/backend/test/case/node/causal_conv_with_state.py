# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_causal_conv_with_state import (
    CausalConvWithState as _RefCausalConvWithState,
)


def _compute(input_, weight, bias=None, past_state=None, activation="none"):
    op = _RefCausalConvWithState.__new__(_RefCausalConvWithState)
    return op._run(
        input_,
        weight,
        bias=bias,
        past_state=past_state,
        activation=activation,
    )


class CausalConvWithState(Base):
    @staticmethod
    def export_basic() -> None:
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)

        output, present_state = _compute(input_, weight)

        expect(
            node,
            inputs=[input_, weight],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_basic",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_with_bias() -> None:
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight", "bias"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)
        bias = np.random.randn(channels).astype(np.float32)

        output, present_state = _compute(input_, weight, bias=bias)

        expect(
            node,
            inputs=[input_, weight, bias],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_with_bias",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_with_past_state() -> None:
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight", "", "past_state"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)
        past_state = np.random.randn(batch_size, channels, k - 1).astype(np.float32)

        output, present_state = _compute(input_, weight, past_state=past_state)

        expect(
            node,
            inputs=[input_, weight, past_state],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_with_past_state",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_silu() -> None:
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight"],
            outputs=["output", "present_state"],
            activation="silu",
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)

        output, present_state = _compute(input_, weight, activation="silu")

        expect(
            node,
            inputs=[input_, weight],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_silu",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_swish_alias() -> None:
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight"],
            outputs=["output", "present_state"],
            activation="swish",
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)

        output, present_state = _compute(input_, weight, activation="swish")

        expect(
            node,
            inputs=[input_, weight],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_swish_alias",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_decode_step() -> None:
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight", "bias", "past_state"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 2, 4, 1, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)
        bias = np.random.randn(channels).astype(np.float32)
        past_state = np.random.randn(batch_size, channels, k - 1).astype(np.float32)

        output, present_state = _compute(
            input_, weight, bias=bias, past_state=past_state
        )

        expect(
            node,
            inputs=[input_, weight, bias, past_state],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_decode_step",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_kernel_size_one() -> None:
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 2, 4, 8, 1
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)

        output, present_state = _compute(input_, weight)

        expect(
            node,
            inputs=[input_, weight],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_kernel_size_one",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_with_bias_and_past_state() -> None:
        # Multi-token (T>1) path through Concat(past, input) -> Conv(+bias).
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight", "bias", "past_state"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)
        bias = np.random.randn(channels).astype(np.float32)
        past_state = np.random.randn(batch_size, channels, k - 1).astype(np.float32)

        output, present_state = _compute(
            input_, weight, bias=bias, past_state=past_state
        )

        expect(
            node,
            inputs=[input_, weight, bias, past_state],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_with_bias_and_past_state",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_silu_with_past_state() -> None:
        # Fused activation combined with concat-from-past variant of PaddedInput.
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight", "", "past_state"],
            outputs=["output", "present_state"],
            activation="silu",
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)
        past_state = np.random.randn(batch_size, channels, k - 1).astype(np.float32)

        output, present_state = _compute(
            input_, weight, past_state=past_state, activation="silu"
        )

        expect(
            node,
            inputs=[input_, weight, past_state],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_silu_with_past_state",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_b1_c1_degenerate() -> None:
        # Mamba/GDN inner-head edge case: B=1, C=1.
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 1, 1, 6, 4
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)

        output, present_state = _compute(input_, weight)

        expect(
            node,
            inputs=[input_, weight],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_b1_c1_degenerate",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_short_input_no_past_state() -> None:
        # L < k-1 with no past_state: zero-pad is wider than the input.
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 2, 4, 2, 5
        input_ = np.random.randn(batch_size, channels, length).astype(np.float32)
        weight = np.random.randn(channels, 1, k).astype(np.float32)

        output, present_state = _compute(input_, weight)

        expect(
            node,
            inputs=[input_, weight],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_short_input_no_past_state",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_fp16() -> None:
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight"],
            outputs=["output", "present_state"],
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.rand(batch_size, channels, length).astype(np.float16)
        weight = np.random.rand(channels, 1, k).astype(np.float16)

        output, present_state = _compute(input_, weight)

        expect(
            node,
            inputs=[input_, weight],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_fp16",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )

    @staticmethod
    def export_silu_fp16() -> None:
        # fp16 + SiLU: the reference upcasts Sigmoid/Mul to float32, so the
        # function-body expansion must do the same to stay numerically faithful.
        node = onnx.helper.make_node(
            "CausalConvWithState",
            inputs=["input", "weight"],
            outputs=["output", "present_state"],
            activation="silu",
        )

        batch_size, channels, length, k = 2, 4, 8, 4
        input_ = np.random.rand(batch_size, channels, length).astype(np.float16)
        weight = np.random.rand(channels, 1, k).astype(np.float16)

        output, present_state = _compute(input_, weight, activation="silu")

        expect(
            node,
            inputs=[input_, weight],
            outputs=[output, present_state],
            name="test_causal_conv_with_state_silu_fp16",
            opset_imports=[onnx.helper.make_opsetid("", 27)],
        )
