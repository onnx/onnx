# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorTriluOffset(unittest.TestCase):
    def _check_trilu(self, data, k=None, upper=None):
        feeds = {"data": data}
        if k is not None:
            feeds["k"] = np.array(k, dtype=np.int64)
        attrs = {} if upper is None else {"upper": upper}
        node = helper.make_node("Trilu", list(feeds), ["output"], **attrs)
        infos = [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in feeds.items()
        ]
        model = helper.make_model(
            helper.make_graph(
                [node],
                "trilu_offset",
                infos,
                [
                    helper.make_tensor_value_info(
                        "output",
                        helper.np_dtype_to_tensor_dtype(data.dtype),
                        data.shape,
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", 14)],
        )
        checker.check_model(model)
        expected = np.zeros_like(data)
        diagonal = 0 if k is None else k
        is_upper = upper is None or upper != 0
        for row in range(data.shape[-2]):
            for column in range(data.shape[-1]):
                keep = (
                    column - row >= diagonal if is_upper else column - row <= diagonal
                )
                if keep:
                    expected[..., row, column] = data[..., row, column]
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, data.dtype)
        self.assertEqual(actual.shape, data.shape)

    def test_extreme_int64_offsets(self):
        for shape in ((2, 3), (2, 3, 2)):
            for dtype in (np.float32, np.int64, np.bool_):
                data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
                for k in (-(2**63), -(2**63) + 1, 2**63 - 1):
                    for upper in (0, 1):
                        with self.subTest(shape=shape, dtype=dtype, k=k, upper=upper):
                            self._check_trilu(data, k, upper)

    def test_diagonal_boundaries_and_empty_shapes(self):
        for shape in ((2, 3), (3, 2), (2, 2, 3), (0, 3), (3, 0), (0, 2, 3)):
            data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            rows, columns = shape[-2:]
            for k in (
                -rows - 1,
                -rows,
                -rows + 1,
                0,
                columns - 1,
                columns,
                columns + 1,
            ):
                for upper in (0, 1):
                    with self.subTest(shape=shape, k=k, upper=upper):
                        self._check_trilu(data, k, upper)

    def test_omitted_k_and_upper(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        for upper in (None, 0, 1):
            with self.subTest(upper=upper):
                self._check_trilu(data, upper=upper)


if __name__ == "__main__":
    unittest.main()
