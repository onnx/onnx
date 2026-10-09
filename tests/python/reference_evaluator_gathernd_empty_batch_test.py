# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorGatherNDEmptyBatch(unittest.TestCase):
    def _check_gathernd(self, opset, data, indices, batch_dims, expected):
        self.assertLess(batch_dims, min(data.ndim, indices.ndim))
        self.assertEqual(data.shape[:batch_dims], indices.shape[:batch_dims])
        self.assertGreaterEqual(indices.shape[-1], 1)
        self.assertLessEqual(indices.shape[-1], data.ndim - batch_dims)
        node = helper.make_node(
            "GatherND", ["data", "indices"], ["output"], batch_dims=batch_dims
        )
        feeds = {"data": data, "indices": indices}
        infos = [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in feeds.items()
        ]
        model = helper.make_model(
            helper.make_graph(
                [node],
                "gathernd_empty_batch",
                infos,
                [
                    helper.make_tensor_value_info(
                        "output",
                        helper.np_dtype_to_tensor_dtype(data.dtype),
                        expected.shape,
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", opset)],
        )
        checker.check_model(model, full_check=True)
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, data.dtype)
        self.assertEqual(actual.shape, expected.shape)

    def test_empty_batch_dimensions(self):
        cases = (
            ((0, 2), (0, 1), 1, (0,)),
            ((0, 3, 2), (0, 4, 1), 1, (0, 4, 2)),
            ((2, 0, 3), (2, 0, 1), 2, (2, 0)),
            ((0, 2, 3), (0, 4, 2), 1, (0, 4)),
            ((2, 0, 3, 4), (2, 0, 5, 1), 2, (2, 0, 5, 4)),
        )
        for opset in (12, 13):
            for dtype in (np.float32, np.int64, np.bool_, object):
                for data_shape, index_shape, batch_dims, output_shape in cases:
                    with self.subTest(
                        opset=opset,
                        dtype=dtype,
                        data_shape=data_shape,
                        batch_dims=batch_dims,
                    ):
                        self._check_gathernd(
                            opset,
                            np.empty(data_shape, dtype=dtype),
                            np.empty(index_shape, dtype=np.int64),
                            batch_dims,
                            np.empty(output_shape, dtype=dtype),
                        )

    def test_empty_indices_with_nonempty_batches(self):
        for opset in (12, 13):
            for dtype in (np.float32, np.int64, np.bool_, object):
                for index_shape, batch_dims, output_shape in (
                    ((0, 1), 0, (0, 3)),
                    ((2, 0, 1), 1, (2, 0)),
                ):
                    with self.subTest(opset=opset, dtype=dtype, batch_dims=batch_dims):
                        values = (
                            [["a", "b", "c"], ["d", "e", "f"]]
                            if dtype is object
                            else [[0, 1, 2], [3, 4, 5]]
                        )
                        self._check_gathernd(
                            opset,
                            np.array(values, dtype=dtype),
                            np.empty(index_shape, dtype=np.int64),
                            batch_dims,
                            np.empty(output_shape, dtype=dtype),
                        )

    def test_nonempty_gathers(self):
        for opset in (12, 13):
            for dtype in (np.float32, np.int64, np.bool_, object):
                data = np.arange(8).reshape(2, 2, 2)
                if dtype is object:
                    data = data.astype(str)
                data = data.astype(dtype)
                for indices, batch_dims, expected in (
                    (
                        np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int64),
                        0,
                        data[[0, 1], [0, 1], [0, 1]],
                    ),
                    (np.array([[1], [0]], dtype=np.int64), 1, data[[0, 1], [1, 0]]),
                ):
                    with self.subTest(opset=opset, dtype=dtype, batch_dims=batch_dims):
                        self._check_gathernd(opset, data, indices, batch_dims, expected)


if __name__ == "__main__":
    unittest.main()
