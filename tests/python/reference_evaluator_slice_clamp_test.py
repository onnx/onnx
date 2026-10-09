# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorSliceClamp(unittest.TestCase):
    def _check_slice(self, opset, data, starts, ends, axes, steps, expected):
        feeds = {"data": data, "starts": starts, "ends": ends}
        inputs = ["data", "starts", "ends"]
        if axes is not None:
            feeds["axes"] = axes
            inputs.append("axes")
        elif steps is not None:
            inputs.append("")
        if steps is not None:
            feeds["steps"] = steps
            inputs.append("steps")
        node = helper.make_node("Slice", inputs, ["output"])
        infos = [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in feeds.items()
        ]
        model = helper.make_model(
            helper.make_graph(
                [node],
                "slice_clamp",
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
        checker.check_model(model)
        original_starts = starts.copy()
        actual = ReferenceEvaluator(model).run(None, feeds)[0]
        assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, data.dtype)
        assert_array_equal(starts, original_starts)

    def test_negative_step_start_below_dimension(self):
        data = np.array([10, 20, 30, 40], dtype=np.float32)
        expected = np.array([10], dtype=data.dtype)
        for opset in (10, 11, 13):
            for dtype in (np.int32, np.int64):
                for axis in (None, 0, -1):
                    for start in (-5, np.iinfo(dtype).min):
                        with self.subTest(
                            opset=opset, dtype=dtype, axis=axis, start=start
                        ):
                            self._check_slice(
                                opset,
                                data,
                                np.array([start], dtype=dtype),
                                np.array([-5], dtype=dtype),
                                None if axis is None else np.array([axis], dtype=dtype),
                                np.array([-1], dtype=dtype),
                                expected,
                            )

    def test_multiple_axes_with_mixed_steps(self):
        data = np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int64)
        expected = np.array([[10], [50]], dtype=data.dtype)
        for opset in (10, 11, 13):
            for dtype in (np.int32, np.int64):
                for axes, starts, ends, steps in (
                    (None, [0, -5], [2, -5], [1, -2]),
                    ([-1, -2], [-5, 0], [-5, 2], [-2, 1]),
                ):
                    with self.subTest(opset=opset, dtype=dtype, axes=axes):
                        self._check_slice(
                            opset,
                            data,
                            np.array(starts, dtype=dtype),
                            np.array(ends, dtype=dtype),
                            None if axes is None else np.array(axes, dtype=dtype),
                            np.array(steps, dtype=dtype),
                            expected,
                        )

    def test_existing_slice_boundaries(self):
        data = np.array([10, 20, 30, 40], dtype=np.float32)
        cases = (
            (-4, -5, -1, [10]),
            (-3, -5, -1, [20, 10]),
            (-1, -5, -1, [40, 30, 20, 10]),
            (100, -5, -1, [40, 30, 20, 10]),
            (-5, 0, -1, []),
            (-5, 4, 1, [10, 20, 30, 40]),
            (-5, 4, None, [10, 20, 30, 40]),
            (0, 4, 2, [10, 30]),
        )
        for opset in (10, 11, 13):
            for dtype in (np.int32, np.int64):
                for start, end, step, values in cases:
                    with self.subTest(opset=opset, dtype=dtype, start=start, step=step):
                        self._check_slice(
                            opset,
                            data,
                            np.array([start], dtype=dtype),
                            np.array([end], dtype=dtype),
                            None,
                            None if step is None else np.array([step], dtype=dtype),
                            np.array(values, dtype=data.dtype),
                        )


if __name__ == "__main__":
    unittest.main()
