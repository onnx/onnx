# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import TensorProto, checker
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
    np_dtype_to_tensor_dtype,
)
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorMatMulInteger:
    @pytest.mark.parametrize("a_dtype", [np.int8, np.uint8])
    @pytest.mark.parametrize("b_dtype", [np.int8, np.uint8])
    @pytest.mark.parametrize("shape", [(2, 2, 2), (2, 3, 4), (1, 3, 2)])
    @pytest.mark.parametrize("zero_point_kind", ["row", "scalar", "omitted"])
    def test_zero_points(self, a_dtype, b_dtype, shape, zero_point_kind):
        m, k, n = shape
        a = np.arange(m * k, dtype=a_dtype).reshape(m, k)
        b = np.arange(k * n, dtype=b_dtype).reshape(k, n)
        feeds = {"A": a, "B": b}
        if zero_point_kind == "row":
            az = np.arange(1, m + 1, dtype=a_dtype)
            bz = np.arange(1, n + 1, dtype=b_dtype)
            feeds.update(az=az, bz=bz)
        elif zero_point_kind == "scalar":
            az = np.array(2, dtype=a_dtype)
            bz = np.array(3, dtype=b_dtype)
            feeds.update(az=az, bz=bz)
        else:
            az = np.array(0, dtype=a_dtype)
            bz = np.array(0, dtype=b_dtype)

        expected = np.array(
            [
                [
                    sum(
                        (int(a[i, t]) - int(az[i] if az.ndim else az))
                        * (int(b[t, j]) - int(bz[j] if bz.ndim else bz))
                        for t in range(k)
                    )
                    for j in range(n)
                ]
                for i in range(m)
            ],
            dtype=np.int32,
        )
        model = make_model(
            make_graph(
                [make_node("MatMulInteger", list(feeds), ["Y"])],
                "matmul_integer",
                [
                    make_tensor_value_info(
                        name, np_dtype_to_tensor_dtype(value.dtype), value.shape
                    )
                    for name, value in feeds.items()
                ],
                [make_tensor_value_info("Y", TensorProto.INT32, [m, n])],
            ),
            opset_imports=[make_opsetid("", 10)],
        )
        checker.check_model(model, full_check=True)
        (got,) = ReferenceEvaluator(model).run(None, feeds)

        assert got.dtype == np.int32
        assert_array_equal(got, expected)

    @pytest.mark.parametrize("b_is_vector", [False, True])
    def test_row_zero_points_broadcast(self, b_is_vector):
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        az = np.array([1, 3], dtype=np.uint8)
        if b_is_vector:
            b = np.array([1, 2, 3], dtype=np.int8)
        else:
            b = np.arange(24, dtype=np.int8).reshape(2, 3, 4)
        node = make_node("MatMulInteger", ["A", "B", "az"], ["Y"])
        expected = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32) @ b

        (got,) = ReferenceEvaluator(node).run(None, {"A": a, "B": b, "az": az})

        assert got.dtype == np.int32
        assert_array_equal(got, expected)

    def test_batched_zero_points(self):
        a = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        b = np.arange(24, dtype=np.int8).reshape(2, 3, 4)
        az = np.array([[[1], [3]], [[2], [4]]], dtype=np.uint8)
        bz = np.array([[[1, 2, 3, 4]], [[4, 3, 2, 1]]], dtype=np.int8)
        node = make_node("MatMulInteger", ["A", "B", "az", "bz"], ["Y"])
        expected = (a.astype(np.int32) - az) @ (b.astype(np.int32) - bz)

        (got,) = ReferenceEvaluator(node).run(
            None, {"A": a, "B": b, "az": az, "bz": bz}
        )

        assert got.dtype == np.int32
        assert_array_equal(got, expected)
