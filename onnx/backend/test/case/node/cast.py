# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools

import numpy as np

import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import (
    make_tensor,
    tensor_dtype_to_np_dtype,
)
from onnx.numpy_helper import to_float8e8m0

F8_TYPES = frozenset({"FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ"})
FOUR_BIT_TYPES = frozenset({"UINT4", "INT4", "FLOAT4E2M1"})


class Cast(Base):
    @staticmethod
    def export() -> None:
        test_cases = [
            ("FLOAT", "FLOAT16"),
            ("FLOAT", "DOUBLE"),
            ("FLOAT16", "FLOAT"),
            ("FLOAT16", "DOUBLE"),
            ("DOUBLE", "FLOAT"),
            ("DOUBLE", "FLOAT16"),
            ("FLOAT", "BFLOAT16"),
            ("BFLOAT16", "FLOAT"),
            ("FLOAT", "FLOAT8E4M3FN"),
            ("FLOAT16", "FLOAT8E4M3FN"),
            ("FLOAT", "FLOAT8E4M3FNUZ"),
            ("FLOAT16", "FLOAT8E4M3FNUZ"),
            ("FLOAT8E4M3FN", "FLOAT"),
            ("FLOAT8E4M3FN", "FLOAT16"),
            ("FLOAT8E4M3FNUZ", "FLOAT"),
            ("FLOAT8E4M3FNUZ", "FLOAT16"),
            ("FLOAT", "FLOAT8E5M2"),
            ("FLOAT16", "FLOAT8E5M2"),
            ("FLOAT", "FLOAT8E5M2FNUZ"),
            ("FLOAT16", "FLOAT8E5M2FNUZ"),
            ("FLOAT8E5M2", "FLOAT"),
            ("FLOAT8E5M2", "FLOAT16"),
            ("FLOAT8E5M2FNUZ", "FLOAT"),
            ("FLOAT8E5M2FNUZ", "FLOAT16"),
            ("FLOAT", "UINT4"),
            ("FLOAT16", "UINT4"),
            ("FLOAT", "INT4"),
            ("FLOAT16", "INT4"),
            ("UINT4", "FLOAT"),
            ("UINT4", "FLOAT16"),
            ("UINT4", "UINT8"),
            ("INT4", "FLOAT"),
            ("INT4", "FLOAT16"),
            ("INT4", "INT8"),
            ("FLOAT4E2M1", "FLOAT"),
            ("FLOAT4E2M1", "FLOAT16"),
            ("FLOAT", "FLOAT4E2M1"),
            ("FLOAT16", "FLOAT4E2M1"),
        ]

        for from_type, to_type in test_cases:
            if from_type == to_type:
                # Skip cases where from_type and to_type are the same
                continue
            from_dtype = getattr(TensorProto, from_type)
            to_dtype = getattr(TensorProto, to_type)
            from_np_dtype = tensor_dtype_to_np_dtype(from_dtype)
            to_np_dtype = tensor_dtype_to_np_dtype(to_dtype)

            if from_type == "BFLOAT16" or to_type == "BFLOAT16":
                np_fp32 = np.array(
                    [
                        "0.47892547",
                        "0.48033667",
                        "0.49968487",
                        "0.81910545",
                        "0.47031248",
                        "0.816468",
                        "0.21087195",
                        "0.7229038",
                        "NaN",
                        "INF",
                        "+INF",
                        "-INF",
                    ],
                    dtype=np.float32,
                )
                input_shape = (3, 4)

            elif from_type in F8_TYPES or to_type in F8_TYPES:
                np_fp32 = np.array(
                    [
                        "0.47892547",
                        "0.48033667",
                        "0.49968487",
                        "0.81910545",
                        "0.47031248",
                        "0.7229038",
                        "1000000",
                        "1e-7",
                        "NaN",
                        "INF",
                        "+INF",
                        "-INF",
                        "-0.0000001",
                        "0.0000001",
                        "-1000000",
                    ],
                    dtype=np.float32,
                )
                input_shape = (3, 5)
            elif from_type in ("UINT4", "INT4") or to_type in ("UINT4", "INT4"):
                np_fp32 = np.arange(-9, 16).astype(np.float32)
                input_shape = (5, 5)
            elif from_type == "FLOAT4E2M1" or to_type == "FLOAT4E2M1":
                np_fp32 = np.array(
                    [
                        "0.48",
                        "0.25",
                        "1.05",
                        "-3.5",
                        "-8",
                        "9",
                        "1000000",
                        "1e-7",
                        "NaN",
                        "INF",
                        "+INF",
                        "-INF",
                        "-4",
                        "0.01",
                        "-0.0",
                    ],
                    dtype=np.float32,
                )
                input_shape = (3, 5)

            else:
                np_fp32 = np.array(
                    [
                        "0.47892547",
                        "0.48033667",
                        "0.49968487",
                        "0.81910545",
                        "0.47031248",
                        "0.816468",
                        "0.21087195",
                        "0.7229038",
                        "NaN",
                        "INF",
                        "+INF",
                        "-INF",
                    ],
                    dtype=np.float32,
                ).reshape([3, 4])
                input_shape = (3, 4)

            if from_type in F8_TYPES:
                np_from = onnx.numpy_helper.saturate_cast(np_fp32, from_np_dtype)
                input = make_tensor(
                    "input",
                    from_dtype,
                    input_shape,
                    vals=np_from,
                    raw=True,
                )
            elif from_type in FOUR_BIT_TYPES:
                np_from = np_fp32.astype(from_np_dtype)
                packed = onnx.numpy_helper._pack_4bitx2(np_from)
                input = make_tensor(
                    "input", from_dtype, input_shape, vals=packed.tobytes(), raw=True
                )
            else:
                np_from = np_fp32.astype(from_np_dtype)
                input = make_tensor(
                    "input", from_dtype, input_shape, vals=np_from, raw=True
                )

            if to_type in F8_TYPES:
                output = make_tensor(
                    "output",
                    to_dtype,
                    input_shape,
                    vals=onnx.numpy_helper.saturate_cast(np_from, to_np_dtype),
                    raw=True,
                )
            elif to_type in FOUR_BIT_TYPES:
                packed = onnx.numpy_helper._pack_4bitx2(np_from.astype(to_np_dtype))
                output = make_tensor(
                    "output", to_dtype, input_shape, vals=packed.tobytes(), raw=True
                )
            else:
                output = make_tensor(
                    "output",
                    to_dtype,
                    input_shape,
                    vals=np_from.astype(to_np_dtype),
                    raw=True,
                )

            node = onnx.helper.make_node(
                "Cast",
                inputs=["input"],
                outputs=["output"],
                to=to_dtype,
            )
            expect(
                node,
                inputs=[input],
                outputs=[output],
                name="test_cast_" + from_type + "_to_" + to_type,
            )

    @staticmethod
    def export_saturate_false() -> None:
        test_cases = itertools.product(
            [
                "FLOAT",
                "FLOAT16",
            ],
            [
                "FLOAT8E4M3FN",
                "FLOAT8E4M3FNUZ",
                "FLOAT8E5M2",
                "FLOAT8E5M2FNUZ",
            ],
        )
        input_shape = (3, 5)
        for from_type, to_type in test_cases:
            from_dtype = getattr(TensorProto, from_type)
            to_dtype = getattr(TensorProto, to_type)
            from_np_dtype = tensor_dtype_to_np_dtype(from_dtype)
            to_np_dtype = tensor_dtype_to_np_dtype(to_dtype)
            np_fp32 = np.array(
                [
                    "0.47892547",
                    "0.48033667",
                    "0.49968487",
                    "0.81910545",
                    "0.47031248",
                    "0.7229038",
                    "1000000",
                    "1e-7",
                    "NaN",
                    "INF",
                    "+INF",
                    "-INF",
                    "-0.0000001",
                    "0.0000001",
                    "-1000000",
                ],
                dtype=np.float32,
            )

            input = make_tensor(
                "input",
                from_dtype,
                input_shape,
                vals=np_fp32.astype(from_np_dtype),
                raw=True,
            )
            output = make_tensor(
                "output",
                to_dtype,
                input_shape,
                vals=np_fp32.astype(from_np_dtype).astype(to_np_dtype),
                raw=True,
            )

            node = onnx.helper.make_node(
                "Cast",
                inputs=["input"],
                outputs=["output"],
                to=to_dtype,
                saturate=0,
            )
            expect(
                node,
                inputs=[input],
                outputs=[output],
                name="test_cast_no_saturate_" + from_type + "_to_" + to_type,
            )

    @staticmethod
    def export_e8m0() -> None:
        np_fp32 = np.array(
            [
                "0.0",
                "0.124",
                "0.25",
                "0.5",
                "1.1",
                "2.0",
                "4.0",
                "8.0",
            ],
            dtype=np.float32,
        )
        test_cases = [
            ("FLOAT", "FLOAT8E8M0"),
            ("FLOAT16", "FLOAT8E8M0"),
            ("FLOAT8E8M0", "FLOAT"),
            ("FLOAT8E8M0", "FLOAT16"),
        ]
        for from_type, to_type in test_cases:
            if from_type == "FLOAT":
                input_np = np_fp32
                output_np = to_float8e8m0(np_fp32)
            elif from_type == "FLOAT16":
                input_np = np_fp32.astype(np.float16)
                output_np = to_float8e8m0(input_np)
            elif from_type == "FLOAT8E8M0":
                input_np = to_float8e8m0(np_fp32)
                if to_type == "FLOAT":
                    output_np = input_np.astype(np.float32)
                elif to_type == "FLOAT16":
                    output_np = input_np.astype(np.float16)
                else:
                    raise ValueError(
                        f"Conversion from {from_type} to {to_type} is not tested."
                    )
            else:
                raise ValueError(
                    f"Conversion from {from_type} to {to_type} is not tested."
                )
            input = make_tensor(
                "input",
                getattr(TensorProto, from_type),
                [2, 4],
                input_np.tobytes(),
                raw=True,
            )
            output = make_tensor(
                "output",
                getattr(TensorProto, to_type),
                [2, 4],
                output_np.tobytes(),
                raw=True,
            )
            if to_type == "FLOAT8E8M0":
                node = onnx.helper.make_node(
                    "Cast",
                    inputs=["input"],
                    outputs=["output"],
                    to=getattr(TensorProto, to_type),
                    saturate=1,
                    round_mode="up",
                )
            else:
                node = onnx.helper.make_node(
                    "Cast",
                    inputs=["input"],
                    outputs=["output"],
                    to=getattr(TensorProto, to_type),
                )

            expect(
                node,
                inputs=[input],
                outputs=[output],
                name="test_cast_e8m0_" + from_type + "_to_" + to_type,
            )
