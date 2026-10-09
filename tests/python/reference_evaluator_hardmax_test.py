# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_hardmax import Hardmax


class TestHardmax(unittest.TestCase):
    def test_hardmax_opset_axis_semantics(self):
        x = np.array([[[1, 4], [3, 2]], [[5, 5], [1, 0]]], dtype=np.float32)
        cases = [
            (1, None, [[[0, 1], [0, 0]], [[1, 0], [0, 0]]]),
            (11, None, [[[0, 1], [0, 0]], [[1, 0], [0, 0]]]),
            (11, 1, [[[0, 1], [0, 0]], [[1, 0], [0, 0]]]),
            (11, -2, [[[0, 1], [0, 0]], [[1, 0], [0, 0]]]),
            (11, 0, [[[0, 0], [0, 0]], [[1, 0], [0, 0]]]),
            (11, -1, [[[0, 1], [1, 0]], [[1, 0], [1, 0]]]),
            (13, None, [[[0, 1], [1, 0]], [[1, 0], [1, 0]]]),
            (13, 1, [[[0, 1], [1, 0]], [[1, 1], [0, 0]]]),
        ]
        for dtype, tensor_type in [
            (np.float16, TensorProto.FLOAT16),
            (np.float32, TensorProto.FLOAT),
            (np.float64, TensorProto.DOUBLE),
        ]:
            for opset, axis, expected in cases:
                with self.subTest(dtype=dtype, opset=opset, axis=axis):
                    attributes = {} if axis is None else {"axis": axis}
                    model = helper.make_model(
                        helper.make_graph(
                            [helper.make_node("Hardmax", ["X"], ["Y"], **attributes)],
                            "hardmax",
                            [helper.make_tensor_value_info("X", tensor_type, x.shape)],
                            [helper.make_tensor_value_info("Y", tensor_type, x.shape)],
                        ),
                        opset_imports=[helper.make_opsetid("", opset)],
                    )
                    checker.check_model(model)
                    result = ReferenceEvaluator(model, new_ops=[Hardmax]).run(
                        None, {"X": x.astype(dtype)}
                    )[0]
                    np.testing.assert_array_equal(
                        result, np.array(expected, dtype=dtype)
                    )
                    self.assertEqual(result.dtype, dtype)

    def test_hardmax_empty(self):
        for opset in (11, 13):
            with self.subTest(opset=opset):
                x = np.empty((0, 2, 3), dtype=np.float32)
                model = helper.make_model(
                    helper.make_graph(
                        [helper.make_node("Hardmax", ["X"], ["Y"])],
                        "hardmax_empty",
                        [
                            helper.make_tensor_value_info(
                                "X", TensorProto.FLOAT, x.shape
                            )
                        ],
                        [
                            helper.make_tensor_value_info(
                                "Y", TensorProto.FLOAT, x.shape
                            )
                        ],
                    ),
                    opset_imports=[helper.make_opsetid("", opset)],
                )
                result = ReferenceEvaluator(model, new_ops=[Hardmax]).run(
                    None, {"X": x}
                )[0]
                np.testing.assert_array_equal(result, x)


if __name__ == "__main__":
    unittest.main()
