# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class LogCumSumExp(Base):
    @staticmethod
    def export_logcumsumexp() -> None:
        x = np.array([-200, -100, 100], dtype=np.float32)
        node = onnx.helper.make_node("LogCumSumExp", ["X"], ["Y"])
        expect(node, [x], [x.copy()], name="test_logcumsumexp_dynamic_range")

    @staticmethod
    def export_logcumsumexp_axes() -> None:
        x = np.log(np.arange(1, 25, dtype=np.float64)).reshape(2, 3, 4)
        for axis in (0, 1, 2, -1, -3):
            node = onnx.helper.make_node("LogCumSumExp", ["X"], ["Y"], axis=axis)
            y = np.log(np.cumsum(np.exp(x), axis=axis))
            expect(node, [x], [y], name=f"test_logcumsumexp_axis_{axis}")

    @staticmethod
    def export_logcumsumexp_special_values() -> None:
        x = np.array(
            [
                [-np.inf, -np.inf, -np.inf, -np.inf],
                [-np.inf, 0, np.inf, -np.inf],
                [0, np.inf, np.inf, 1],
                [0, np.nan, np.inf, -np.inf],
                [np.inf, -np.inf, np.nan, 0],
            ],
            dtype=np.float32,
        )
        y = np.array(
            [
                [-np.inf, -np.inf, -np.inf, -np.inf],
                [-np.inf, 0, np.inf, np.inf],
                [0, np.inf, np.inf, np.inf],
                [0, np.nan, np.nan, np.nan],
                [np.inf, np.inf, np.nan, np.nan],
            ],
            dtype=np.float32,
        )
        node = onnx.helper.make_node("LogCumSumExp", ["X"], ["Y"], axis=1)
        expect(node, [x], [y], name="test_logcumsumexp_special_values")

    @staticmethod
    def export_logcumsumexp_empty() -> None:
        for shape, axis in (((2, 0, 3), 1), ((0, 3), -1)):
            x = np.empty(shape, dtype=np.float32)
            node = onnx.helper.make_node("LogCumSumExp", ["X"], ["Y"], axis=axis)
            expect(node, [x], [x.copy()], name=f"test_logcumsumexp_empty_axis_{axis}")

    @staticmethod
    def export_logcumsumexp_low_precision() -> None:
        for dtype in (np.float16, ml_dtypes.bfloat16):
            x = np.zeros(128, dtype=dtype)
            y = np.log(np.arange(1, 129, dtype=np.float64)).astype(dtype)
            node = onnx.helper.make_node("LogCumSumExp", ["X"], ["Y"])
            expect(node, [x], [y], name=f"test_logcumsumexp_{np.dtype(dtype).name}")
