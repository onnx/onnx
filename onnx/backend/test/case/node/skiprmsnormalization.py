# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_skip_rms_normalization import _skip_rms_normalization


class SkipRMSNormalization(Base):
    @staticmethod
    def export_3d() -> None:
        x = np.random.randn(3, 4, 2).astype(np.float32)
        skip = np.random.randn(3, 4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        y, input_skip_bias_sum = _skip_rms_normalization(x, skip, gamma, B=bias)
        y.astype(np.float32)
        input_skip_bias_sum.astype(np.float32)

        node = onnx.helper.make_node(
            "SkipRMSNormalization",
            inputs=["x", "skip", "gamma", "bias"],
            outputs=["y", "input_skip_bias_sum"],
        )

        expect(
            node,
            inputs=[x, skip, gamma, bias],
            outputs=[y, input_skip_bias_sum],
            name="test_skip_rms_normalization_3d_example",
        )

    @staticmethod
    def export_2d() -> None:
        x = np.random.randn(4, 2).astype(np.float32)
        skip = np.random.randn(4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        y, input_skip_bias_sum = _skip_rms_normalization(x, skip, gamma, B=bias)
        y.astype(np.float32)
        input_skip_bias_sum.astype(np.float32)

        node = onnx.helper.make_node(
            "SkipRMSNormalization",
            inputs=["x", "skip", "gamma", "bias"],
            outputs=["y", "input_skip_bias_sum"],
        )

        expect(
            node,
            inputs=[x, skip, gamma, bias],
            outputs=[y, input_skip_bias_sum],
            name="test_skip_rms_normalization_2d_example",
        )

    @staticmethod
    def export_epsilon() -> None:
        x = np.random.randn(3, 4, 2).astype(np.float32)
        skip = np.random.randn(3, 4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        epsilon = 1e-2
        y, input_skip_bias_sum = _skip_rms_normalization(
            x, skip, gamma, B=bias, epsilon=epsilon
        )
        y.astype(np.float32)
        input_skip_bias_sum.astype(np.float32)

        node = onnx.helper.make_node(
            "SkipRMSNormalization",
            inputs=["x", "skip", "gamma", "bias"],
            outputs=["y", "input_skip_bias_sum"],
            epsilon=epsilon,
        )

        expect(
            node,
            inputs=[x, skip, gamma, bias],
            outputs=[y, input_skip_bias_sum],
            name="test_skip_rms_normalization_epsilon_example",
        )

    @staticmethod
    def export_scaling_factor() -> None:
        x = np.random.randn(3, 4, 2).astype(np.float32)
        skip = np.random.randn(3, 4, 2).astype(np.float32)
        gamma = np.random.randn(2).astype(np.float32)
        bias = np.random.randn(2).astype(np.float32)
        scaling_factor = 3
        y, input_skip_bias_sum = _skip_rms_normalization(
            x, skip, gamma, B=bias, scaling_factor=scaling_factor
        )
        y.astype(np.float32)
        input_skip_bias_sum.astype(np.float32)

        node = onnx.helper.make_node(
            "SkipRMSNormalization",
            inputs=["x", "skip", "gamma", "bias"],
            outputs=["y", "input_skip_bias_sum"],
            scaling_factor=scaling_factor,
        )

        expect(
            node,
            inputs=[x, skip, gamma, bias],
            outputs=[y, input_skip_bias_sum],
            name="test_skip_rms_normalization_scaling_factor_example",
        )
