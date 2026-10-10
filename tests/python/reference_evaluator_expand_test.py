# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.op_expand import Expand


class TestExpandValuePreservation(unittest.TestCase):
    def check_expand(self, data, shape, elem_type):
        node = helper.make_node("Expand", ["X", "shape"], ["Y"])
        graph = helper.make_graph(
            [node],
            "expand",
            [
                helper.make_tensor_value_info("X", elem_type, list(data.shape)),
                helper.make_tensor_value_info("shape", TensorProto.INT64, [len(shape)]),
            ],
            [helper.make_tensor_value_info("Y", elem_type, None)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        result = ReferenceEvaluator(model, new_ops=[Expand]).run(
            None, {"X": data, "shape": np.array(shape, dtype=np.int64)}
        )[0]
        expected = np.broadcast_to(data, np.broadcast_shapes(data.shape, tuple(shape)))
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, data.dtype)
        self.assertTrue(result.flags.writeable)
        self.assertFalse(np.shares_memory(result, data))
        return result

    def test_complex_nonfinite_values(self):
        for dtype, elem_type in [
            (np.complex64, TensorProto.COMPLEX64),
            (np.complex128, TensorProto.COMPLEX128),
        ]:
            for value in [
                complex(float("inf"), 0),
                complex(0, float("inf")),
                complex(float("-inf"), 2),
                complex(1, 2),
            ]:
                with self.subTest(dtype=dtype, value=value):
                    self.check_expand(np.array([value], dtype=dtype), [2, 1], elem_type)

    def test_string_values(self):
        for dtype in [np.str_, np.bytes_, object]:
            with self.subTest(dtype=dtype):
                self.check_expand(
                    np.array(["one", "two"], dtype=dtype), [3, 2], TensorProto.STRING
                )

    def test_multidirectional_shapes(self):
        self.check_expand(
            np.array([[1], [2]], dtype=np.int32), [1, 3], TensorProto.INT32
        )
        self.check_expand(np.array(3, dtype=np.int32), [2, 3], TensorProto.INT32)
        self.check_expand(np.empty((0, 1), dtype=np.float32), [1, 3], TensorProto.FLOAT)
        self.check_expand(np.array([[True], [False]]), [1, 3], TensorProto.BOOL)

    def test_invalid_broadcast(self):
        with self.assertRaises(ValueError):
            self.check_expand(
                np.ones((2, 3), dtype=np.float32), [4, 3], TensorProto.FLOAT
            )


if __name__ == "__main__":
    unittest.main()
