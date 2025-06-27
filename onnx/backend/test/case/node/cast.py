# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import ml_dtypes
import numpy as np

import itertools
import onnx
from onnx import TensorProto, helper, subbyte
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import (
    tensor_dtype_to_np_dtype,
    make_tensor,
    tensor_dtype_to_field,
)
from onnx.numpy_helper import (
    _float8e4m3_to_float32,
    _float8e5m2_to_float32,
    _unpacked_float4e2m1_to_float32,
)
from onnx.subbyte import _float32_to_float4e2m1_unpacked


class Cast(Base):
    @staticmethod
    def export() -> None:
        shape = (3, 4)
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
        test_cases = itertools.product(
            all_numeric_dtypes, all_numeric_dtypes
        )

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
                ).reshape([3, 4])
                input_shape = (3, 4)

            if from_type in f8_types:
                input = make_tensor(
                    "input", from_dtype, input_shape, vals=onnx.numpy_helper.saturating_cast(np_fp32, from_np_dtype)
                )
            else:
                input = make_tensor(
                    "input", from_dtype, input_shape, vals=np_fp32.astype(from_np_dtype)
                )
            if to_type in f8_types:
                output = make_tensor(
                    "output", to_dtype, input_shape, vals=onnx.numpy_helper.saturating_cast(np_fp32, to_np_dtype)
                )
            else:
                output = make_tensor(
                    "output", to_dtype, input_shape, vals=np_fp32.astype(from_np_dtype).astype(to_np_dtype)
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
        test_cases = [
            ("FLOAT", "FLOAT8E4M3FN"),
            ("FLOAT16", "FLOAT8E4M3FN"),
            ("FLOAT", "FLOAT8E4M3FNUZ"),
            ("FLOAT16", "FLOAT8E4M3FNUZ"),
            ("FLOAT", "FLOAT8E5M2"),
            ("FLOAT16", "FLOAT8E5M2"),
            ("FLOAT", "FLOAT8E5M2FNUZ"),
            ("FLOAT16", "FLOAT8E5M2FNUZ"),
        ]
        vect_float32_to_float8e4m3 = np.vectorize(_float32_to_float8e4m3)
        vect_float32_to_float8e5m2 = np.vectorize(_float32_to_float8e5m2)

        for from_type, to_type in test_cases:
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

            if from_type == "FLOAT":
                input_values = np_fp32
                input = make_tensor("x", TensorProto.FLOAT, [3, 5], np_fp32.tolist())
            elif from_type == "FLOAT16":
                input_values = np_fp32.astype(np.float16).astype(np.float32)
                input = make_tensor(
                    "x", TensorProto.FLOAT16, [3, 5], input_values.tolist()
                )
            else:
                raise ValueError(
                    f"Conversion from {from_type} to {to_type} is not tested."
                )

            if to_type == "FLOAT8E4M3FN":
                expected = vect_float32_to_float8e4m3(input_values, saturate=False)
            elif to_type == "FLOAT8E4M3FNUZ":
                expected = vect_float32_to_float8e4m3(
                    input_values, uz=True, saturate=False
                )
            elif to_type == "FLOAT8E5M2":
                expected = vect_float32_to_float8e5m2(input_values, saturate=False)
            elif to_type == "FLOAT8E5M2FNUZ":
                expected = vect_float32_to_float8e5m2(
                    input_values, fn=True, uz=True, saturate=False
                )
            else:
                raise ValueError(
                    f"Conversion from {from_type} to {to_type} is not tested."
                )

            ivals = bytes([int(i) for i in expected])
            tensor = TensorProto()
            tensor.data_type = getattr(TensorProto, to_type)
            tensor.name = "x"
            tensor.dims.extend([3, 5])
            field = tensor_dtype_to_field(tensor.data_type)
            getattr(tensor, field).extend(ivals)

            output = tensor

            node = onnx.helper.make_node(
                "Cast",
                inputs=["input"],
                outputs=["output"],
                to=getattr(TensorProto, to_type),
                saturate=0,
            )
            expect(
                node,
                inputs=[input],
                outputs=[output],
                name="test_cast_no_saturate_" + from_type + "_to_" + to_type,
            )
