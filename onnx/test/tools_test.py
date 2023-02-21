# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from numpy.testing import assert_allclose
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.tools import update_model_dims
from onnx.tools.replace_constants import replace_initializer_by_constant_of_shape


class TestToolsFunctions(unittest.TestCase):
    def test_update_inputs_outputs_dim(self) -> None:
        node_def = helper.make_node(
            "Conv",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
        )
        graph_def = helper.make_graph(
            [node_def],
            "test",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 5]),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])],
        )
        model_def = helper.make_model(graph_def, producer_name="test")
        updated_def = update_model_dims.update_inputs_outputs_dims(
            model_def,
            {
                "x": [1, 1, "x1", -1],
                "W": [1, 1, 3, 3],
            },
            {
                "y": [1, 1, -1, -1],
            },
        )
        onnx.checker.check_model(updated_def)
        self.assertEqual(
            updated_def.graph.input[0].type.tensor_type.shape.dim[2].dim_param, "x1"
        )
        self.assertEqual(
            updated_def.graph.input[0].type.tensor_type.shape.dim[3].dim_param, "x_3"
        )
        self.assertEqual(
            updated_def.graph.output[0].type.tensor_type.shape.dim[2].dim_param, "y_2"
        )
        self.assertEqual(
            updated_def.graph.output[0].type.tensor_type.shape.dim[3].dim_param, "y_3"
        )

    def test_replace_initializer(self):
        dtype = np.float32
        value = np.random.randn(2, 100).astype(dtype)
        A = numpy_helper.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = numpy_helper.from_array(value, name="C")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = helper.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = helper.make_node("Sub", ["AX", "C"], ["Y"])
        graph = helper.make_graph([node1, node2], "lr", [X], [Y], [A, C])
        model_def = helper.make_model(graph)

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]
        repl = replace_initializer_by_constant_of_shape(model_def)
        node_types = set(n.op_type for n in repl.graph.node)
        self.assertIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y1[:, :] = 3.5
        y1[0, :] = 0.5
        y2 = oinf2.run(None, {"X": x})[0]
        assert_allclose(y1, y2)


if __name__ == "__main__":
    unittest.main()
