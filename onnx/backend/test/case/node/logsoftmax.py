# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops._op_common_softmax import _log_softmax


class LogSoftmax(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
        )
        x = np.array([[-1, 0, 1]]).astype(np.float32)
        # expected output
        # [[-2.4076061 -1.407606  -0.407606 ]]
        y = _log_softmax(x)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_example_1")

    @staticmethod
    def export_logsoftmax_axis() -> None:
        x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
        # expected output
        # [[-3.4401896  -2.4401896  -1.4401896  -0.44018966]
        # [-3.4401896  -2.4401896  -1.4401896  -0.44018966]]
        y = _log_softmax(x)

        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
        )
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_large_number")

        x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
            axis=0,
        )
        y = _log_softmax(x, axis=0)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_axis_0")

        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
            axis=1,
        )
        y = _log_softmax(x, axis=1)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_axis_1")

        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
            axis=2,
        )
        y = _log_softmax(x, axis=2)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_axis_2")

        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
            axis=-1,
        )
        y = _log_softmax(x, axis=-1)
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_negative_axis")

        # default axis is -1
        node = onnx.helper.make_node(
            "LogSoftmax",
            inputs=["x"],
            outputs=["y"],
        )
        expect(node, inputs=[x], outputs=[y], name="test_logsoftmax_default_axis")

    @staticmethod
    def export_logsoftmax_small_log_probability() -> None:
        for dtype, margin in [
            (np.float16, 10.0),
            (np.float32, 18.0),
            (np.float64, 40.0),
        ]:
            x = np.full((3, 3), -margin, dtype=dtype)
            np.fill_diagonal(x, 0)
            x = np.stack([x, x])
            # Compute the expected value analytically, independently of the helper.
            normalizer = np.log1p(2 * np.exp(-margin))
            expected = (x.astype(np.float64) - normalizer).astype(dtype)
            for axis in [0, 1, -1]:
                node = onnx.helper.make_node("LogSoftmax", ["x"], ["y"], axis=axis)
                expect(
                    node,
                    inputs=[np.moveaxis(x, -1, axis)],
                    outputs=[np.moveaxis(expected, -1, axis)],
                    name=f"test_logsoftmax_small_log_probability_{np.dtype(dtype).name}_axis_{axis}",
                    rtol=4 * float(np.finfo(dtype).eps),
                    atol=0,
                )

    @staticmethod
    def export_logsoftmax_tied_maxima() -> None:
        node = onnx.helper.make_node("LogSoftmax", ["x"], ["y"], axis=-1)
        x = np.array([[0, 0, -200], [4, 4, 4], [0, -np.inf, -np.inf]], dtype=np.float64)
        expected = np.array(
            [
                [-np.log(2), -np.log(2), -200 - np.log(2)],
                [-np.log(3), -np.log(3), -np.log(3)],
                [0, -np.inf, -np.inf],
            ],
            dtype=np.float64,
        )
        expect(
            node,
            inputs=[x],
            outputs=[expected],
            name="test_logsoftmax_tied_maxima",
            rtol=1e-15,
            atol=0,
        )

    @staticmethod
    def export_logsoftmax_single_class() -> None:
        node = onnx.helper.make_node("LogSoftmax", ["x"], ["y"], axis=1)
        x = np.array([[[1, -2, 3]], [[4, 5, -6]]], dtype=np.float32)
        expect(
            node,
            inputs=[x],
            outputs=[np.zeros_like(x)],
            name="test_logsoftmax_single_class",
            atol=0,
        )

    @staticmethod
    def export_logsoftmax_empty_input() -> None:
        node = onnx.helper.make_node("LogSoftmax", ["x"], ["y"], axis=1)
        x = np.empty((0, 3), dtype=np.float32)
        expect(node, inputs=[x], outputs=[x], name="test_logsoftmax_empty_input")

    @staticmethod
    def export_logsoftmax_large_finite_gap() -> None:
        node = onnx.helper.make_node("LogSoftmax", ["x"], ["y"], axis=1)
        x = np.array([[0, -200]], dtype=np.float32)
        expect(
            node,
            inputs=[x],
            outputs=[x],
            name="test_logsoftmax_large_finite_gap",
            rtol=0,
            atol=0,
        )
