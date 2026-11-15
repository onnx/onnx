# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import ml_dtypes
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Expm1(Base):
    @staticmethod
    def export_expm1() -> None:
        node = onnx.helper.make_node("Expm1", ["x"], ["y"])
        for dtype in [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]:
            x = np.array([[-10.0, -0.5, 0.0, 0.25], [0.5, 1.0, 3.0, 5.0]], dtype=dtype)
            y = np.array([math.expm1(float(v)) for v in x.flat], dtype=dtype).reshape(
                x.shape
            )
            expect(
                node,
                inputs=[x],
                outputs=[y],
                name=f"test_expm1_{np.dtype(dtype).name}",
                rtol=1e-2 if dtype == ml_dtypes.bfloat16 else 1e-3,
            )

    @staticmethod
    def export_expm1_small_inputs() -> None:
        node = onnx.helper.make_node("Expm1", ["x"], ["y"])
        for dtype, tiny in [
            (np.float16, 2.0**-14),
            (ml_dtypes.bfloat16, 2.0**-10),
            (np.float32, 1e-8),
            (np.float64, 1e-20),
        ]:
            x = np.array([-tiny, -0.0, 0.0, tiny], dtype=dtype)
            y = np.array([math.expm1(float(v)) for v in x], dtype=dtype)
            # Relative error allows backend variation, while atol=0 catches
            # an incorrect zero caused by rounding the intermediate exp(x).
            expect(
                node,
                inputs=[x],
                outputs=[y],
                name=f"test_expm1_small_inputs_{np.dtype(dtype).name}",
                rtol=1e-2 if dtype == ml_dtypes.bfloat16 else 1e-3,
                atol=0,
            )

    @staticmethod
    def export_expm1_special_values() -> None:
        node = onnx.helper.make_node("Expm1", ["x"], ["y"])
        for dtype in [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]:
            x = np.array([-0.0, 0.0, -np.inf, np.inf, np.nan], dtype=dtype)
            y = np.array([-0.0, 0.0, -1.0, np.inf, np.nan], dtype=dtype)
            expect(
                node,
                inputs=[x],
                outputs=[y],
                name=f"test_expm1_special_values_{np.dtype(dtype).name}",
            )

    @staticmethod
    def export_expm1_scalar_and_empty() -> None:
        node = onnx.helper.make_node("Expm1", ["x"], ["y"])
        expect(
            node,
            inputs=[np.array(1.0, dtype=np.float32)],
            outputs=[np.array(math.expm1(1), dtype=np.float32)],
            name="test_expm1_scalar",
        )
        x = np.empty((0, 3), dtype=np.float32)
        expect(node, inputs=[x], outputs=[x], name="test_expm1_empty")
