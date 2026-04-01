# SPDX-License-Identifier: Apache-2.0

# Copyright (c) ONNX Project Contributors
from __future__ import annotations

import unittest

import parameterized

import onnx.helper
import onnx.shape_inference


class NodeInferenceTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("GreaterOrEqual",),
            ("LessOrEqual",),
        ]
    )
    def test_comparison_op(self, op_type):
        node = onnx.helper.make_node(op_type, ["x", "y"], ["z"])
        schema = onnx.defs.get_schema(node.op_type, 23, "")
        xtype = onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT32, [1, 10])
        ytype = onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT32, [10, 1])
        result = onnx.shape_inference.infer_node_outputs(
            schema, node, {"x": xtype, "y": ytype}
        )
        self.assertEqual(list(result.keys()), ["z"])
        self.assertEqual(result["z"].tensor_type.elem_type, onnx.TensorProto.BOOL)
        self.assertEqual(
            [dim.dim_value for dim in result["z"].tensor_type.shape.dim],
            [10, 10],
        )


if __name__ == "__main__":
    unittest.main()
