# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect

# Expected outputs are literals computed in float64 with math.erf and math.tanh,
# then rounded to the output type, so they do not depend on the reference
# implementation. The gate values include the negative tail (-3, -4), where Gelu
# is most sensitive to the precision it is computed in.
_A = np.array(
    [[[1.0, -2.0, 3.0], [4.0, -1.0, 0.5]], [[-3.0, 2.0, -0.5], [-4.0, 0.0, 1.5]]],
    dtype=np.float32,
)
_B = np.array(
    [[[0.5, 1.0, -1.0], [2.0, 2.0, -1.0]], [[0.5, 1.0, 1.5], [-2.0, 3.0, 0.25]]],
    dtype=np.float32,
)
_Y_NONE = np.array(
    [
        [[0.4206724, -0.045500264, -2.9959502], [7.999747, -0.3173105, -0.34573123]],
        [[-0.002024847, 1.9544997, -0.23140316], [0.00025336994, 0.0, 0.3499473]],
    ],
    dtype=np.float32,
)

_A_2D = [[-3.0, -4.0, -1.0, 2.0], [-0.5, 1.5, -2.5, 0.75]]
_B_2D = [[0.5, 1.0, -1.0, 2.0], [2.0, -1.0, 0.5, 1.0]]


class GeGLU(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
        )
        expect(
            node,
            inputs=[_A, _B],
            outputs=[_Y_NONE],
            name="test_geglu",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_approximate_none() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
            approximate="none",
        )
        expect(
            node,
            inputs=[_A, _B],
            outputs=[_Y_NONE],
            name="test_geglu_approximate_none",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_tanh() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
            approximate="tanh",
        )
        y = np.array(
            [
                [
                    [0.420596, -0.045402307, -2.9963627],
                    [7.9998593, -0.31761602, -0.345714],
                ],
                [
                    [-0.001818696, 1.9545977, -0.23142898],
                    [0.0001404919, 0.0, 0.34989288],
                ],
            ],
            dtype=np.float32,
        )
        expect(
            node,
            inputs=[_A, _B],
            outputs=[y],
            name="test_geglu_tanh",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_float16() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
        )
        a = np.array(_A_2D, dtype=np.float16)
        b = np.array(_B_2D, dtype=np.float16)
        y = np.array(
            [
                [
                    -0.002025604248046875,
                    -0.00012671947479248047,
                    0.15869140625,
                    3.908203125,
                ],
                [-0.30859375, -1.3994140625, -0.007762908935546875, 0.580078125],
            ],
            dtype=np.float16,
        )
        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_geglu_float16",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_bfloat16() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
        )
        a = np.array(_A_2D, dtype=ml_dtypes.bfloat16)
        b = np.array(_B_2D, dtype=ml_dtypes.bfloat16)
        y = np.array(
            [
                [-0.0020294189453125, -0.00012683868408203125, 0.158203125, 3.90625],
                [-0.30859375, -1.3984375, -0.00775146484375, 0.578125],
            ],
            dtype=ml_dtypes.bfloat16,
        )
        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_geglu_bfloat16",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )

    @staticmethod
    def export_double() -> None:
        node = onnx.helper.make_node(
            "GeGLU",
            inputs=["a", "b"],
            outputs=["y"],
        )
        a = np.array(_A_2D, dtype=np.float64)
        b = np.array(_B_2D, dtype=np.float64)
        y = np.array(
            [
                [
                    -0.002024847047445155,
                    -0.00012668496733247991,
                    0.15865525393145707,
                    3.908999472207283,
                ],
                [
                    -0.3085375387259869,
                    -1.399789198096713,
                    -0.007762081657220199,
                    0.5800294857173488,
                ],
            ],
            dtype=np.float64,
        )
        expect(
            node,
            inputs=[a, b],
            outputs=[y],
            name="test_geglu_double",
            opset_imports=[onnx.helper.make_opsetid("", 29)],
        )
