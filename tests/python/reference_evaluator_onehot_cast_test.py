# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_one_hot import OneHot


class TestOneHotCast(unittest.TestCase):
    def _check(self, indices, depth, expected, axis=-1):
        values = np.array([-2, 7], dtype=np.int32)
        expected = np.where(np.array(expected, dtype=bool), values[1], values[0])
        if axis == 0:
            expected = expected.T
        inputs = {"indices": indices, "depth": depth, "values": values}
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("OneHot", list(inputs), ["Y"], axis=axis)],
                "onehot_cast",
                [
                    helper.make_tensor_value_info(
                        name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
                    )
                    for name, value in inputs.items()
                ],
                [
                    helper.make_tensor_value_info(
                        "Y",
                        helper.np_dtype_to_tensor_dtype(values.dtype),
                        expected.shape,
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", 11)],
        )
        checker.check_model(model)
        actual = ReferenceEvaluator(model, new_ops=[OneHot]).run(None, inputs)[0]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, values.dtype)

    def test_fractional_indices(self):
        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        for dtype in (np.float16, np.float32, np.float64):
            for axis in (-1, 0):
                with self.subTest(dtype=dtype, axis=axis):
                    self._check(
                        np.array(
                            [0.9, 1.9, 2.1, -0.9, -1.9, -3.9, -4.1, 3.9], dtype=dtype
                        ),
                        np.array(3, dtype=np.int64),
                        expected,
                        axis,
                    )

    def test_fractional_depth(self):
        indices = np.array([0, 1, 2, 3, -1, -3, -4], dtype=np.int64)
        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 0, 0],
        ]
        for dtype in (np.float16, np.float32, np.float64):
            for depth in (np.array(3.8, dtype=dtype), np.array([3.8], dtype=dtype)):
                for axis in (-1, 0):
                    with self.subTest(dtype=dtype, depth_shape=depth.shape, axis=axis):
                        self._check(indices, depth, expected, axis)

    def test_integer_controls(self):
        self._check(
            np.array([0, 2, 2**64 - 1], dtype=np.uint64),
            np.array(3, dtype=np.int64),
            [[1, 0, 0], [0, 0, 1], [0, 0, 0]],
        )
        self._check(
            np.array([], dtype=np.int64),
            np.array(3, dtype=np.int64),
            np.empty((0, 3), dtype=bool),
        )


if __name__ == "__main__":
    unittest.main()
