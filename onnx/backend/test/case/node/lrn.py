# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class LRN(Base):
    @staticmethod
    def export() -> None:
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        nsize = 3
        node = onnx.helper.make_node(
            "LRN",
            inputs=["x"],
            outputs=["y"],
            alpha=alpha,
            beta=beta,
            bias=bias,
            size=nsize,
        )
        x = np.random.randn(1, 5, 5, 5).astype(np.float32)
        square_sum = np.zeros_like(x)
        for n, c, h, w in np.ndindex(x.shape):
            square_sum[n, c, h, w] = sum(
                x[
                    n,
                    max(0, c - math.floor((nsize - 1) / 2)) : min(
                        x.shape[1], c + math.ceil((nsize - 1) / 2) + 1
                    ),
                    h,
                    w,
                ]
                ** 2
            )
        y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
        expect(node, inputs=[x], outputs=[y], name="test_lrn")

    @staticmethod
    def export_default() -> None:
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        nsize = 3
        node = onnx.helper.make_node("LRN", inputs=["x"], outputs=["y"], size=3)
        x = np.random.randn(1, 5, 5, 5).astype(np.float32)
        square_sum = np.zeros_like(x)
        for n, c, h, w in np.ndindex(x.shape):
            square_sum[n, c, h, w] = sum(
                x[
                    n,
                    max(0, c - math.floor((nsize - 1) / 2)) : min(
                        x.shape[1], c + math.ceil((nsize - 1) / 2) + 1
                    ),
                    h,
                    w,
                ]
                ** 2
            )
        y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
        expect(node, inputs=[x], outputs=[y], name="test_lrn_default")

    @staticmethod
    def export_float16_large_values() -> None:
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        nsize = 5
        node = onnx.helper.make_node("LRN", inputs=["x"], outputs=["y"], size=nsize)
        x = np.full((1, 5, 1, 1), 300).astype(np.float16)
        # 300 ** 2 overflows float16 (max 65504), so compute in float32.
        x32 = x.astype(np.float32)
        square_sum = np.zeros_like(x32)
        for c in range(x.shape[1]):
            begin = max(0, c - math.floor((nsize - 1) / 2))
            end = min(x.shape[1], c + math.ceil((nsize - 1) / 2) + 1)
            square_sum[:, c] = np.sum(x32[:, begin:end] ** 2, axis=1)
        y = (x32 / ((bias + (alpha / nsize) * square_sum) ** beta)).astype(np.float16)
        expect(node, inputs=[x], outputs=[y], name="test_lrn_float16_large_values")
