# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class QLinearMatMul(Base):
    @staticmethod
    def export_int() -> None:
        for quant_type_name in ["uint8", "int8"]:
            quant_type = getattr(np, quant_type_name)
            for dtype_name in ["float32", "float16"]:
                dtype = getattr(np, dtype_name)
                node = onnx.helper.make_node(
                    "QLinearMatMul",
                    inputs=[
                        "a",
                        "a_scale",
                        "a_zero_point",
                        "b",
                        "b_scale",
                        "b_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    outputs=["y"],
                )

                # 2D
                a = np.array([[208, 236, 0, 238], [3, 214, 255, 29]])
                if quant_type == np.int8:
                    a -= 127
                a = a.astype(quant_type)

                a_scale = np.array([0.0066], dtype=dtype)
                a_zero_point = np.array(
                    [113 - 127] if quant_type == np.int8 else [113], dtype=quant_type
                )

                b = np.array(
                    [[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]]
                )
                if quant_type == np.int8:
                    b -= 127
                b = b.astype(quant_type)

                b_scale = np.array([0.00705], dtype=dtype)
                b_zero_point = np.array(
                    [114 - 127] if quant_type == np.int8 else [114], dtype=quant_type
                )

                y_scale = np.array([0.0107], dtype=dtype)
                y_zero_point = np.array(
                    [118 - 127] if quant_type == np.int8 else [118], dtype=quant_type
                )

                if quant_type == np.int8:
                    output = np.array([[41, -12, -9], [1, -75, 20]])
                else:
                    output = np.array([[168, 115, 255], [1, 66, 151]])
                output = output.astype(quant_type)

                expect(
                    node,
                    inputs=[
                        a,
                        a_scale,
                        a_zero_point,
                        b,
                        b_scale,
                        b_zero_point,
                        y_scale,
                        y_zero_point,
                    ],
                    outputs=[output],
                    name=f"test_qlinearmatmul_2D_{quant_type_name}_{dtype_name}",
                )

                # 3D
                a = np.array(
                    [
                        [[208, 236, 0, 238], [3, 214, 255, 29]],
                        [[208, 236, 0, 238], [3, 214, 255, 29]],
                    ],
                )
                if quant_type == np.int8:
                    a -= 127
                a = a.astype(quant_type)

                a_scale = np.array([0.0066], dtype=dtype)
                a_zero_point = np.array(
                    [113 - 127] if quant_type == np.int8 else [113], dtype=quant_type
                )

                b = np.array(
                    [
                        [[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]],
                        [[152, 51, 244], [60, 26, 255], [0, 127, 246], [127, 254, 247]],
                    ],
                )
                if quant_type == np.int8:
                    b -= 127
                b = b.astype(quant_type)

                b_scale = np.array([0.00705], dtype=dtype)
                b_zero_point = np.array([114], dtype=quant_type)

                y_scale = np.array([0.0107], dtype=dtype)
                y_zero_point = np.array(
                    [118 - 127] if quant_type == np.int8 else [118], dtype=quant_type
                )

                if quant_type == np.int8:
                    if dtype == np.float32:
                        output = np.array(
                            [
                                [[-86, 117, 120], [115, 39, -121]],
                                [[-86, 117, 120], [115, 39, -121]],
                            ]
                        )
                    else:
                        output = np.array(
                            [
                                [[-86, 116, 119], [115, 39, -121]],
                                [[-86, 116, 119], [115, 39, -121]],
                            ]
                        )
                else:
                    output = np.array(
                        [
                            [[168, 115, 255], [1, 66, 151]],
                            [[168, 115, 255], [1, 66, 151]],
                        ]
                    )
                output = output.astype(quant_type)

                expect(
                    node,
                    inputs=[
                        a,
                        a_scale,
                        a_zero_point,
                        b,
                        b_scale,
                        b_zero_point,
                        y_scale,
                        y_zero_point,
                    ],
                    outputs=[output],
                    name=f"test_qlinearmatmul_3D_{quant_type_name}_{dtype_name}",
                )

    @staticmethod
    def export_overflow() -> None:
        """Test clipping behavior when matmul result exceeds output dtype max for uint8 and int8 cases."""
        node = onnx.helper.make_node(
            "QLinearMatMul",
            inputs=[
                "a",
                "a_scale",
                "a_zero_point",
                "b",
                "b_scale",
                "b_zero_point",
                "y_scale",
                "y_zero_point",
            ],
            outputs=["y"],
        )

        # uint8 overflow case
        a = np.array([[100]], dtype=np.uint8)
        b = np.array([[100]], dtype=np.uint8)
        a_scale = np.array([1.0], dtype=np.float32)
        a_zero_point = np.array([0], dtype=np.uint8)
        b_scale = np.array([1.0], dtype=np.float32)
        b_zero_point = np.array([0], dtype=np.uint8)
        y_scale = np.array([0.2], dtype=np.float32)
        y_zero_point = np.array([0], dtype=np.uint8)
        # 100 * 100 / 0.2 = 50000 → clipped to 255
        expected = np.array([[255]], dtype=np.uint8)

        expect(
            node,
            inputs=[
                a,
                a_scale,
                a_zero_point,
                b,
                b_scale,
                b_zero_point,
                y_scale,
                y_zero_point,
            ],
            outputs=[expected],
            name="test_qlinearmatmul_overflow_uint8",
        )

        # int8 overflow case
        a = np.array([[100]], dtype=np.int8)
        b = np.array([[100]], dtype=np.int8)
        a_scale = np.array([1.0], dtype=np.float32)
        a_zero_point = np.array([0], dtype=np.int8)
        b_scale = np.array([1.0], dtype=np.float32)
        b_zero_point = np.array([0], dtype=np.int8)
        y_scale = np.array([0.5], dtype=np.float32)
        y_zero_point = np.array([0], dtype=np.int8)
        # 100 * 100 / 0.5 = 20000 → clipped to 127
        expected = np.array([[127]], dtype=np.int8)

        expect(
            node,
            inputs=[
                a,
                a_scale,
                a_zero_point,
                b,
                b_scale,
                b_zero_point,
                y_scale,
                y_zero_point,
            ],
            outputs=[expected],
            name="test_qlinearmatmul_overflow_int8",
        )

    @staticmethod
    def export_underflow() -> None:
        """Test clipping behavior when matmul result falls below output dtype min for uint8 and int8 cases."""
        node = onnx.helper.make_node(
            "QLinearMatMul",
            inputs=[
                "a",
                "a_scale",
                "a_zero_point",
                "b",
                "b_scale",
                "b_zero_point",
                "y_scale",
                "y_zero_point",
            ],
            outputs=["y"],
        )

        # uint8 underflow case
        a = np.array([[-100]], dtype=np.uint8)
        b = np.array([[100]], dtype=np.uint8)
        a_scale = np.array([1.0], dtype=np.float32)
        a_zero_point = np.array([0], dtype=np.uint8)
        b_scale = np.array([1.0], dtype=np.float32)
        b_zero_point = np.array([0], dtype=np.uint8)
        y_scale = np.array([1.0], dtype=np.float32)
        y_zero_point = np.array([0], dtype=np.uint8)
        # D = -100 * 100 = -10000 → clipped to 0
        expected = np.array([[0]], dtype=np.uint8)

        expect(
            node,
            inputs=[
                a,
                a_scale,
                a_zero_point,
                b,
                b_scale,
                b_zero_point,
                y_scale,
                y_zero_point,
            ],
            outputs=[expected],
            name="test_qlinearmatmul_underflow_uint8",
        )

        # int8 underflow: -100*100 = -10000, scaled with y_scale=0.5 gives -20000 → clips to -128
        a = np.array([[-100]], dtype=np.int8)
        b = np.array([[100]], dtype=np.int8)
        a_scale = np.array([1.0], dtype=np.float32)
        a_zero_point = np.array([0], dtype=np.int8)
        b_scale = np.array([1.0], dtype=np.float32)
        b_zero_point = np.array([0], dtype=np.int8)
        y_scale = np.array([0.5], dtype=np.float32)
        y_zero_point = np.array([0], dtype=np.int8)
        # -100 * 100 / 0.5 = -20000 → clipped to -128
        expected = np.array([[-128]], dtype=np.int8)

        expect(
            node,
            inputs=[
                a,
                a_scale,
                a_zero_point,
                b,
                b_scale,
                b_zero_point,
                y_scale,
                y_zero_point,
            ],
            outputs=[expected],
            name="test_qlinearmatmul_underflow_int8",
        )
