# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from onnx import TensorProto, checker, helper, shape_inference
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops import load_op
from onnx.reference.ops.op_flatten import Flatten

OPSETS = (1, 9, 11, 13, 21, 23, 24, 25)
NEGATIVE_AXIS_OPSETS = OPSETS[2:]


class TestReferenceEvaluatorFlattenBoundary(unittest.TestCase):
    def _check_flatten(self, shape, axis, expected_shape, opset):
        # These are schema-legal axes. Older schemas do not support negative axes.
        if axis is not None:
            self.assertGreaterEqual(
                axis, -len(shape) if opset in NEGATIVE_AXIS_OPSETS else 0
            )
            self.assertLessEqual(axis, len(shape))
        attributes = {} if axis is None else {"axis": axis}
        node = helper.make_node("Flatten", ["X"], ["Y"], **attributes)
        graph = helper.make_graph(
            [node],
            "flatten_boundary",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])],
        )
        model = helper.make_model_gen_version(
            graph, opset_imports=[helper.make_opsetid("", opset)]
        )
        checker.check_model(model, full_check=True)
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
        dimensions = inferred.graph.output[0].type.tensor_type.shape.dim
        self.assertTrue(all(dim.HasField("dim_value") for dim in dimensions))
        self.assertEqual(tuple(dim.dim_value for dim in dimensions), expected_shape)

        data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        evaluator = ReferenceEvaluator(model)
        self.assertIs(load_op("", "Flatten", opset), Flatten)
        self.assertIs(type(evaluator.rt_nodes_[0]), Flatten)
        (actual,) = evaluator.run(None, {"X": data})
        self.assertEqual(actual.shape, expected_shape)
        self.assertEqual(actual.dtype, data.dtype)
        assert_array_equal(actual, data.reshape(expected_shape))

    def test_empty_dimensions_nonnegative_axis(self):
        cases = (
            ((0,), 0, (1, 0)),
            ((0,), 1, (0, 1)),
            ((0, 3), 0, (1, 0)),
            ((0, 3), 1, (0, 3)),
            ((0, 3), 2, (0, 1)),
            ((2, 0, 3), 0, (1, 0)),
            ((2, 0, 3), 1, (2, 0)),
            ((2, 0, 3), 2, (0, 3)),
            ((2, 0, 3), 3, (0, 1)),
            ((2, 3, 0), 1, (2, 0)),
            ((2, 3, 0), 2, (6, 0)),
            ((2, 3, 0), 3, (0, 1)),
            ((0, 2, 0, 3), 1, (0, 0)),
            ((0, 2, 0, 3), 3, (0, 3)),
        )
        for opset in OPSETS:
            for shape, axis, expected_shape in cases:
                with self.subTest(opset=opset, shape=shape, axis=axis):
                    self._check_flatten(shape, axis, expected_shape, opset)

    def test_empty_dimensions_negative_axis(self):
        cases = (
            ((0,), -1, (1, 0)),
            ((0, 3), -2, (1, 0)),
            ((0, 3), -1, (0, 3)),
            ((2, 0, 3), -3, (1, 0)),
            ((2, 0, 3), -2, (2, 0)),
            ((2, 0, 3), -1, (0, 3)),
            ((2, 3, 0), -1, (6, 0)),
            ((0, 2, 0, 3), -1, (0, 3)),
            ((0, 2, 0, 3), -3, (0, 0)),
        )
        for opset in NEGATIVE_AXIS_OPSETS:
            for shape, axis, expected_shape in cases:
                with self.subTest(opset=opset, shape=shape, axis=axis):
                    self._check_flatten(shape, axis, expected_shape, opset)

    def test_nonempty_axis_endpoints(self):
        for opset in OPSETS:
            for axis, expected_shape in ((0, (1, 24)), (3, (24, 1))):
                with self.subTest(opset=opset, axis=axis):
                    self._check_flatten((2, 3, 4), axis, expected_shape, opset)

    def test_nonempty_negative_axis(self):
        for opset in NEGATIVE_AXIS_OPSETS:
            for axis, expected_shape in (
                (-3, (1, 24)),
                (-2, (2, 12)),
                (-1, (6, 4)),
            ):
                with self.subTest(opset=opset, axis=axis):
                    self._check_flatten((2, 3, 4), axis, expected_shape, opset)

    def test_empty_dimensions_default_axis(self):
        for opset in OPSETS:
            for shape, expected_shape in (((0, 3), (0, 3)), ((2, 0, 3), (2, 0))):
                with self.subTest(opset=opset, shape=shape):
                    self._check_flatten(shape, None, expected_shape, opset)

    def test_nonempty_default_axis(self):
        for opset in OPSETS:
            with self.subTest(opset=opset):
                self._check_flatten((2, 3, 4), None, (2, 12), opset)


if __name__ == "__main__":
    unittest.main()
