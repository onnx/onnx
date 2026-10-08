# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorMaxPoolCeilBoundary(unittest.TestCase):
    def _check_case(
        self, case_id, x, attributes, expected_y, expected_indices, with_indices
    ):
        output_shape = [1, 1] + [None] * (x.ndim - 2)
        outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)]
        if with_indices:
            outputs.append(
                helper.make_tensor_value_info("I", TensorProto.INT64, output_shape)
            )
        node = helper.make_node(
            "MaxPool", ["X"], [output.name for output in outputs], **attributes
        )
        graph = helper.make_graph(
            [node],
            case_id,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x.shape))],
            outputs,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
        checker.check_model(model, full_check=True)
        evaluator = ReferenceEvaluator(model)
        actual = evaluator.run(None, {"X": x})
        assert_array_equal(actual[0], expected_y)
        self.assertEqual(actual[0].dtype, np.dtype(np.float32))
        if with_indices:
            assert_array_equal(actual[1], expected_indices)
            self.assertEqual(actual[1].dtype, np.dtype(np.int64))

    def test_ceil_mode_excludes_all_right_starting_windows(self):
        # MaxPool-22 permits nonnegative pads and ignores every window whose
        # starting point lies in the right padding. Expected values are literal.
        cases = [
            (
                "one_right_start_1d",
                [[[1, 2]]],
                {"kernel_shape": [2], "strides": [2], "pads": [0, 1], "ceil_mode": 1},
                [[[2]]],
                [[[1]]],
            ),
            (
                "multiple_right_starts_1d",
                [[[1, 2]]],
                {"kernel_shape": [2], "strides": [2], "pads": [0, 4], "ceil_mode": 1},
                [[[2]]],
                [[[1]]],
            ),
            (
                "partial_bottom_multiple_right_starts_2d",
                [[[[1, 2], [3, 4], [5, 6]]]],
                {
                    "kernel_shape": [2, 2],
                    "strides": [2, 2],
                    "pads": [0, 0, 0, 4],
                    "ceil_mode": 1,
                },
                [[[[4], [6]]]],
                [[[[3], [5]]]],
            ),
            (
                "multiple_bottom_and_right_starts_2d",
                [[[[1, 2], [3, 4]]]],
                {
                    "kernel_shape": [2, 2],
                    "strides": [2, 2],
                    "pads": [0, 0, 4, 4],
                    "ceil_mode": 1,
                },
                [[[[4]]]],
                [[[[3]]]],
            ),
        ]
        for case_id, data, attributes, expected_y, expected_indices in cases:
            for with_indices in (False, True):
                with self.subTest(case=case_id, with_indices=with_indices):
                    self._check_case(
                        case_id,
                        np.array(data, dtype=np.float32),
                        attributes,
                        np.array(expected_y, dtype=np.float32),
                        np.array(expected_indices, dtype=np.int64),
                        with_indices,
                    )

    def test_floor_mode_controls(self):
        cases = [
            (
                "floor_1d",
                [[[1, 2]]],
                {"kernel_shape": [2], "strides": [2], "pads": [0, 1], "ceil_mode": 0},
                [[[2]]],
                [[[1]]],
            ),
            (
                "floor_2d",
                [[[[1, 2], [3, 4], [5, 6]]]],
                {
                    "kernel_shape": [2, 2],
                    "strides": [2, 2],
                    "pads": [0, 0, 0, 0],
                    "ceil_mode": 0,
                },
                [[[[4]]]],
                [[[[3]]]],
            ),
        ]
        for case_id, data, attributes, expected_y, expected_indices in cases:
            with self.subTest(case=case_id, with_indices=True):
                self._check_case(
                    case_id,
                    np.array(data, dtype=np.float32),
                    attributes,
                    np.array(expected_y, dtype=np.float32),
                    np.array(expected_indices, dtype=np.int64),
                    True,
                )
