# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from onnx import TensorProto
from onnx.helper import make_node, make_tensor, make_tensor_value_info
# TODO: remove the following ignore after mypy upgrade in ONNX
from shape_inference_test import TestShapeInferenceHelper  # type: ignore
import unittest


class TestDataPropagation(TestShapeInferenceHelper):

    def test_expand_symbolic_input(self) -> None:
        graph = self._make_graph(
            [('x', TensorProto.INT32, (3, 1, 2)),
             ('y', TensorProto.INT32, (1, 4, 2))],
            [make_node("Shape", ['y'], ['shape']),
             make_node("Expand", ['x', 'shape'], ['z'])],
            [])
        self._assert_inferred(graph, [
            make_tensor_value_info('shape', TensorProto.INT64, (3,)),
            make_tensor_value_info('z', TensorProto.INT32, (3, 4, 2))],
            data_prop=True)

    def test_constantofshape_with_symbolic_shape(self) -> None:
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5))],
            [make_node("Shape", ['x'], ['shape']),
             make_node("ConstantOfShape", ['shape'], ['y'], value=make_tensor('value', TensorProto.INT32, (1, ), (2, )))],
            [])
        self._assert_inferred(graph,
            [make_tensor_value_info('shape', TensorProto.INT64, (3,)),
             make_tensor_value_info('y', TensorProto.INT32, (3, 4, 5))], data_prop=True)  # type: ignore


if __name__ == '__main__':
    unittest.main()
