# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator

_ZEROS = [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]
_WEIGHTED = [
    [[0.0, 0.0], [0.6931471805599453, 0.6931471805599453]],
    [[0.6931471805599453, 0.6931471805599453], [0.0, 0.0]],
]
_QUARTERS = [[[0.25, 0.25], [0.25, 0.25]], [[0.25, 0.25], [0.25, 0.25]]]
_HALVES = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]
_LEGACY_WEIGHTED = [
    [[1 / 6, 1 / 6], [1 / 3, 1 / 3]],
    [[1 / 3, 1 / 3], [1 / 6, 1 / 6]],
]
_AXIS_WEIGHTED = [
    [[1 / 3, 1 / 3], [2 / 3, 2 / 3]],
    [[2 / 3, 2 / 3], [1 / 3, 1 / 3]],
]
_CASES = [
    ("opset11_axis1_uniform", 11, 1, _ZEROS, _QUARTERS),
    ("opset11_negative2_uniform", 11, -2, _ZEROS, _QUARTERS),
    ("opset11_axis1_weighted", 11, 1, _WEIGHTED, _LEGACY_WEIGHTED),
    ("opset11_negative2_weighted", 11, -2, _WEIGHTED, _LEGACY_WEIGHTED),
    ("opset11_default_weighted", 11, None, _WEIGHTED, _LEGACY_WEIGHTED),
    ("opset11_axis2_control", 11, 2, _WEIGHTED, _HALVES),
    ("opset11_negative1_control", 11, -1, _WEIGHTED, _HALVES),
    ("opset13_axis1_control", 13, 1, _WEIGHTED, _AXIS_WEIGHTED),
    ("opset13_negative2_control", 13, -2, _WEIGHTED, _AXIS_WEIGHTED),
    ("opset13_axis2_control", 13, 2, _WEIGHTED, _HALVES),
    ("opset13_negative1_control", 13, -1, _WEIGHTED, _HALVES),
    ("opset13_default_control", 13, None, _WEIGHTED, _HALVES),
]


class TestReferenceSoftmax(unittest.TestCase):
    def run_case(self, case):
        name, opset, axis, values, expected_values = case
        x = np.array(values, dtype=np.float32)
        original = x.copy()
        attributes = {} if axis is None else {"axis": axis}
        node = helper.make_node("Softmax", ["X"], ["Y"], **attributes)
        model = helper.make_model(
            helper.make_graph(
                [node],
                name,
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2, 2])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 2])],
            ),
            opset_imports=[helper.make_opsetid("", opset)],
        )
        checker.check_model(model, full_check=True)
        actual = ReferenceEvaluator(model).run(None, {"X": x})[0]
        self.assertEqual(actual.shape, (2, 2, 2))
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(x, original)
        # Literal fractions follow exp(0)=1 and exp(log(2))=2. Opset11
        # normalizes the four features after flattening at axis1; opset13
        # normalizes only the two elements along that axis.
        np.testing.assert_allclose(
            actual, np.array(expected_values, dtype=np.float32), rtol=1e-6, atol=1e-7
        )

    def test_rank3_versioned_axes(self):
        for case in _CASES:
            with self.subTest(case=case[0]):
                self.run_case(case)


if __name__ == "__main__":
    unittest.main()
