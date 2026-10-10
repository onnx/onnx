# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from itertools import product

import numpy as np

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorMatMulIntegerZeroPoint(unittest.TestCase):
    def _check(self, a, b, a_zero_point, b_zero_point, expected, opset):
        feeds = {"A": a, "B": b}
        names = ["A", "B"]
        if a_zero_point is not None or b_zero_point is not None:
            names.append("a_zero_point" if a_zero_point is not None else "")
        if a_zero_point is not None:
            feeds["a_zero_point"] = a_zero_point
        if b_zero_point is not None:
            names.append("b_zero_point")
            feeds["b_zero_point"] = b_zero_point
        node = helper.make_node("MatMulInteger", names, ["Y"])
        inputs = [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), list(value.shape)
            )
            for name, value in feeds.items()
        ]
        graph = helper.make_graph(
            [node],
            "matmulinteger_zero_point",
            inputs,
            [
                helper.make_tensor_value_info(
                    "Y", TensorProto.INT32, list(expected.shape)
                )
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        checker.check_model(model, full_check=True)
        originals = {name: value.copy() for name, value in feeds.items()}
        output = ReferenceEvaluator(model).run(None, feeds)[0]
        self.assertEqual(output.dtype, np.dtype(np.int32))
        self.assertEqual(output.shape, expected.shape)
        np.testing.assert_array_equal(output, expected)
        for name, value in feeds.items():
            np.testing.assert_array_equal(value, originals[name])

    def test_per_row_zero_points_square(self):
        for a_dtype, b_dtype, opset, per_column in product(
            (np.int8, np.uint8), (np.int8, np.uint8), (10, 21), (False, True)
        ):
            with self.subTest(
                a_dtype=a_dtype, b_dtype=b_dtype, opset=opset, per_column=per_column
            ):
                a = np.array([[3, 5], [7, 9]], dtype=a_dtype)
                b = np.array([[1, 2], [3, 4]], dtype=b_dtype)
                a_zero_point = np.array([1, 4], dtype=a_dtype)
                b_zero_point = np.array([1, 2], dtype=b_dtype) if per_column else None
                expected = np.array(
                    [[8, 8], [10, 10]] if per_column else [[14, 20], [18, 26]],
                    dtype=np.int32,
                )
                self._check(a, b, a_zero_point, b_zero_point, expected, opset)

    def test_per_row_zero_points_rectangular(self):
        for a_dtype, b_dtype, opset, per_column in product(
            (np.int8, np.uint8), (np.int8, np.uint8), (10, 21), (False, True)
        ):
            with self.subTest(
                a_dtype=a_dtype, b_dtype=b_dtype, opset=opset, per_column=per_column
            ):
                a = np.array([[3, 5, 7], [7, 9, 11]], dtype=a_dtype)
                b = np.array([[1, 2], [3, 4], [5, 6]], dtype=b_dtype)
                a_zero_point = np.array([1, 4], dtype=a_dtype)
                b_zero_point = np.array([1, 2], dtype=b_dtype) if per_column else None
                expected = np.array(
                    [[32, 32], [38, 38]] if per_column else [[44, 56], [53, 68]],
                    dtype=np.int32,
                )
                self._check(a, b, a_zero_point, b_zero_point, expected, opset)

    def test_scalar_default_and_batched_zero_points(self):
        a = np.array([[3, 5], [7, 9]], dtype=np.uint8)
        b = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        cases = [
            (a, b, None, None, [[18, 26], [34, 50]]),
            (
                a,
                b,
                np.array(1, dtype=np.uint8),
                np.array(1, dtype=np.uint8),
                [[8, 14], [16, 30]],
            ),
            (a, b, np.array([1], dtype=np.uint8), None, [[14, 20], [30, 44]]),
            (
                a[None, ...],
                b[None, ...],
                np.array([[[1], [4]]], dtype=np.uint8),
                np.array([[[1, 2]]], dtype=np.uint8),
                [[[8, 8], [10, 10]]],
            ),
        ]
        for index, (data_a, data_b, zero_a, zero_b, expected) in enumerate(cases):
            with self.subTest(case=index):
                self._check(
                    data_a,
                    data_b,
                    zero_a,
                    zero_b,
                    np.array(expected, dtype=np.int32),
                    10,
                )


if __name__ == "__main__":
    unittest.main()
