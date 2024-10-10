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
def _skip_layer_normalization(x, skip, gamma, beta, B, epsilon=1e-5):
    input_skip_sum = x + skip
    input_skip_bias_sum = input_skip_sum + B
    output, mean, inv_std_dev = _layer_normalization(input_skip_bias_sum, gamma, beta, epsilon=epsilon)
    return output


class SkipLayerNormalization(Base):
    @staticmethod
    def export_3d() -> None:
        x = np.random.randn(3, 4, 2).astype(np.float32)
        skip = np.random.randn(3, 4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        beta = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        y = _skip_layer_normalization(x, skip, gamma, beta, bias).astype(np.float32)

        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["x", "skip", "gamma", "beta", "bias"],
            outputs=["y"],
        )

        expect(
            node,
            inputs=[x, skip, gamma, beta, bias],
            outputs=[y],
            name="test_skip_layer_normalization_3d_example",
        )

    @staticmethod
    def export_2d() -> None:
        x = np.random.randn(4, 2).astype(np.float32)
        skip = np.random.randn(4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        beta = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        y = _skip_layer_normalization(x, skip, gamma, beta, bias).astype(np.float32)

        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["x", "skip", "gamma", "beta", "bias"],
            outputs=["y"],
        )

        expect(
            node,
            inputs=[x, skip, gamma, beta, bias],
            outputs=[y],
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
        y = _skip_layer_normalization(x, skip, gamma, beta, bias).astype(np.float32)

        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["x", "skip", "gamma", "beta", "bias"],
            outputs=["y"],
            epsilon=epsilon,
        )

        expect(
            node,
            inputs=[x, skip, gamma, beta, bias],
            outputs=[y],
            name="test_skip_layer_normalization_epsilon_example",
        )
