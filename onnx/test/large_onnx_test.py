# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import unittest
import numpy as np

from onnx import TensorProto, checker
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from onnx.large_onnx import make_large_model, LargeModelProto


class TestLargeOnnx(unittest.TestCase):
    @staticmethod
    def _linear_regression():
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        graph = make_graph(
            [
                make_node("Matmul", ["X", "A"], ["XA"]),
                make_node("Matmul", ["XA", "B"], ["Y"]),
            ],
            "mm",
            [X],
            [Y],
            [
                from_array(np.arange(9).astype(np.float32).reshape((-1, 3)), name="A"),
                from_array(
                    (np.arange(9) * 10).astype(np.float32).reshape((-1, 3)),
                    name="B",
                ),
            ],
        )
        onnx_model = make_model(graph)
        checker.check_model(onnx_model)
        return onnx_model

    def test_large_onnx_no_large_initializer(self):
        model_proto = self._linear_regression()
        large_proto = make_large_model(model_proto.graph)
        assert isinstance(large_proto, LargeModelProto)


if __name__ == "__main__":
    unittest.main(verbosity=2)
