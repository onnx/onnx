# SPDX-License-Identifier: Apache-2.0
import onnx
import unittest
from onnx import helper, parser, GraphProto


class TestBasicFunctions(unittest.TestCase):
    def check_graph(self, graph: GraphProto) -> None:
        self.assertTrue(len(graph.node) == 3)
        self.assertTrue(graph.node[0].op_type == "MatMul")
        self.assertTrue(graph.node[1].op_type == "Add")
        self.assertTrue(graph.node[2].op_type == "Softmax")

    def test_parse_graph(self) -> None:
        input = '''
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul(X, W)
              S = Add(T, B)
              C = Softmax(S)
           }
           '''
        graph = onnx.parser.parse_graph(input)
        self.check_graph(graph)

    def test_parse_model(self) -> None:
        input = '''
           <
             ir_version: 7,
             opset_import: [ "" : 10, "com.microsoft": 1]
           >
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul(X, W)
              S = Add(T, B)
              C = Softmax(S)
           }
           '''
        model = onnx.parser.parse_model(input)
        self.assertTrue(model.ir_version == 7)
        self.assertTrue(len(model.opset_import) == 2)
        self.check_graph(model.graph)

    def test_parse_graph_error(self) -> None:
        input = '''
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul[X, W]
              S = Add(T, B)
              C = Softmax(S)
           }
           '''
        self.assertRaises(onnx.parser.ParseError, lambda: onnx.parser.parse_graph(input))

    def test_parse_model_error(self) -> None:
        input = '''
           <
             ir_version: 7,
             opset_import: [ "" : 10   "com.microsoft": 1]
           >
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul(X, W)
              S = Add(T, B)
              C = Softmax(S)
           }
           '''
        self.assertRaises(onnx.parser.ParseError, lambda: onnx.parser.parse_model(input))
