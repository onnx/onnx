# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.backend.test.case.node.layernormalization import _layer_normalization


# Skip layer normalization's reference implementation
def _skip_layer_normalization(x, skip, gamma, beta, B, axis=-1, epsilon=1e-5, scaling_factor=1):
    input_skip_sum = x + (skip * scaling_factor)
    input_skip_bias_sum = input_skip_sum + B
    output, _, _ = _layer_normalization(input_skip_bias_sum, gamma, beta, epsilon=epsilon, axis=axis)
    return output, input_skip_bias_sum


class SkipLayerNormalization(Base):
    @staticmethod
    def export_3d() -> None:
        x = np.random.randn(3, 4, 2).astype(np.float32)
        skip = np.random.randn(3, 4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        beta = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        y, input_skip_bias_sum = _skip_layer_normalization(x, skip, gamma, beta, bias)
        y.astype(np.float32)
        input_skip_bias_sum.astype(np.float32)

        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["x", "skip", "gamma", "beta", "bias"],
            outputs=["y", "input_skip_bias_sum"],
        )

        expect(
            node,
            inputs=[x, skip, gamma, beta, bias],
            outputs=[y, input_skip_bias_sum],
            name="test_skip_layer_normalization_3d_example",
        )

    @staticmethod
    def export_2d() -> None:
        x = np.random.randn(4, 2).astype(np.float32)
        skip = np.random.randn(4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        beta = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        y, input_skip_bias_sum = _skip_layer_normalization(x, skip, gamma, beta, bias)
        y.astype(np.float32)
        input_skip_bias_sum.astype(np.float32)

        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["x", "skip", "gamma", "beta", "bias"],
            outputs=["y", "input_skip_bias_sum"],
        )

        expect(
            node,
            inputs=[x, skip, gamma, beta, bias],
            outputs=[y, input_skip_bias_sum],
            name="test_skip_layer_normalization_2d_example",
        )

    @staticmethod
    def export_epsilon() -> None:
        x = np.random.randn(3, 4, 2).astype(np.float32)
        skip = np.random.randn(3, 4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        beta = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        epsilon = 1e-2
        y, input_skip_bias_sum = _skip_layer_normalization(x, skip, gamma, beta, bias, epsilon=epsilon)
        y.astype(np.float32)
        input_skip_bias_sum.astype(np.float32)

        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["x", "skip", "gamma", "beta", "bias"],
            outputs=["y", "input_skip_bias_sum"],
            epsilon=epsilon,
        )

        expect(
            node,
            inputs=[x, skip, gamma, beta, bias],
            outputs=[y, input_skip_bias_sum],
            name="test_skip_layer_normalization_epsilon_example",
        )

    @staticmethod
    def export_scaling_factor() -> None:
        x = np.random.randn(3, 4, 2).astype(np.float32)
        skip = np.random.randn(3, 4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        beta = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        scaling_factor = 3
        y, input_skip_bias_sum = _skip_layer_normalization(x, skip, gamma, beta, bias, scaling_factor=scaling_factor)
        y.astype(np.float32)
        input_skip_bias_sum.astype(np.float32)

        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["x", "skip", "gamma", "beta", "bias"],
            outputs=["y", "input_skip_bias_sum"],
            scaling_factor=scaling_factor,
        )

        expect(
            node,
            inputs=[x, skip, gamma, beta, bias],
            outputs=[y, input_skip_bias_sum],
            name="test_skip_layer_normalization_scaling_factor_example",
        )
