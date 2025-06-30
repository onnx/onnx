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


class Cast(Base):
    @staticmethod
    def export() -> None:
        all_numeric_dtypes = [
            "FLOAT",
            "UINT8",
            "INT8",
            "UINT16",
            "INT16",
            "INT32",
            "INT64",
            "BOOL",
            "FLOAT16",
            "DOUBLE",
            "UINT32",
            "UINT64",
            "BFLOAT16",
            "FLOAT8E4M3FN",
            "FLOAT8E4M3FNUZ",
            "FLOAT8E5M2",
            "FLOAT8E5M2FNUZ",
            "UINT4",
            "INT4",
            "FLOAT4E2M1",
        ]
        test_cases = itertools.product(all_numeric_dtypes, all_numeric_dtypes)

        f8_types = {"FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ"}

        for from_type, to_type in test_cases:
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

            elif from_type in f8_types or to_type in f8_types:
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

            if from_type in f8_types:
                input = make_tensor(
                    "x",
                    from_dtype,
                    input_shape,
                    vals=onnx.numpy_helper.saturating_cast(np_fp32, from_np_dtype),
                    raw=True,
                )
            else:
                input = make_tensor(
                    "x", from_dtype, input_shape, vals=np_fp32.astype(from_np_dtype)
                )
            if to_type in f8_types:
                output = make_tensor(
                    "x",
                    to_dtype,
                    input_shape,
                    vals=onnx.numpy_helper.saturating_cast(np_fp32, to_np_dtype),
                    raw=True,
                )
            else:
                output = make_tensor(
                    "x",
                    to_dtype,
                    input_shape,
                    vals=np_fp32.astype(from_np_dtype).astype(to_np_dtype),
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
                "x",
                from_dtype,
                input_shape,
                vals=np_fp32.astype(from_np_dtype),
                raw=True,
            )
            output = make_tensor(
                "x",
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
