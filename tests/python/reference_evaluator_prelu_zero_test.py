# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorPReluZero(unittest.TestCase):
    def _check_prelu(self, opset, data, slope, expected):
        self.assertEqual(np.broadcast_shapes(data.shape, slope.shape), data.shape)
        self.assertEqual(data.dtype, slope.dtype)
        feeds = {"data": data, "slope": slope}
        node = helper.make_node("PRelu", list(feeds), ["output"])
        infos = [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in feeds.items()
        ]
        model = helper.make_model(
            helper.make_graph(
                [node],
                "prelu_zero",
                infos,
                [
                    helper.make_tensor_value_info(
                        "output",
                        helper.np_dtype_to_tensor_dtype(data.dtype),
                        data.shape,
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", opset)],
        )
        checker.check_model(model, full_check=True)
        with np.errstate(invalid="ignore"):
            actual = ReferenceEvaluator(model).run(None, feeds)[0]
        assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, data.dtype)
        self.assertEqual(actual.shape, data.shape)
        if np.issubdtype(data.dtype, np.floating):
            assert_array_equal(np.signbit(actual), np.signbit(expected))

    def test_zero_with_infinite_slope(self):
        for opset in (9, 16):
            for dtype in (np.float16, np.float32, np.float64):
                data = np.array([0.0, -0.0, 1.0, -1.0], dtype=dtype)
                for value in (np.inf, -np.inf):
                    for shape in ((), (4,)):
                        with self.subTest(
                            opset=opset, dtype=dtype, slope=value, shape=shape
                        ):
                            self._check_prelu(
                                opset,
                                data,
                                np.full(shape, value, dtype=dtype),
                                np.array([0.0, -0.0, 1.0, -value], dtype=dtype),
                            )

    def test_zero_with_negative_finite_slope(self):
        for opset in (9, 16):
            for dtype in (np.float16, np.float32, np.float64):
                with self.subTest(opset=opset, dtype=dtype):
                    self._check_prelu(
                        opset,
                        np.array([0.0, -0.0, 1.0, -1.0], dtype=dtype),
                        np.array(-0.5, dtype=dtype),
                        np.array([0.0, -0.0, 1.0, 0.5], dtype=dtype),
                    )

    def test_finite_and_empty_controls(self):
        for opset in (9, 16):
            for dtype in (np.float16, np.float32, np.float64):
                for data, slope, expected in (
                    ([-2.0, -0.0, 0.0, 3.0], 0.25, [-0.5, -0.0, 0.0, 3.0]),
                    ([-2.0, 1.0, 3.0], -0.25, [0.5, 1.0, 3.0]),
                ):
                    with self.subTest(opset=opset, dtype=dtype, slope=slope):
                        self._check_prelu(
                            opset,
                            np.array(data, dtype=dtype),
                            np.array(slope, dtype=dtype),
                            np.array(expected, dtype=dtype),
                        )
                with self.subTest(opset=opset, dtype=dtype, empty=True):
                    self._check_prelu(
                        opset,
                        np.empty((0, 4), dtype=dtype),
                        np.ones((4,), dtype=dtype),
                        np.empty((0, 4), dtype=dtype),
                    )
            for dtype in (np.int32, np.int64, np.uint32, np.uint64):
                with self.subTest(opset=opset, dtype=dtype):
                    if np.issubdtype(dtype, np.signedinteger):
                        data, expected = [-2, 1, 3], [-4, 1, 3]
                    else:
                        data = expected = [0, 1, 3]
                    self._check_prelu(
                        opset,
                        np.array(data, dtype=dtype),
                        np.array(2, dtype=dtype),
                        np.array(expected, dtype=dtype),
                    )


if __name__ == "__main__":
    unittest.main()
