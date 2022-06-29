# SPDX-License-Identifier: Apache-2.0
import onnx
import unittest
from onnx import helper, parser, GraphProto
from onnx import checker


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
        self.assertRaises(onnx.parser.ParseError,
                          lambda: onnx.parser.parse_graph(input))

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

    def test_parse_function_with_attributes(self) -> None:
        input = '''
            <
            ir_version: 9,
            opset_import: [ "" : 15, "custom_domain" : 1],
            producer_name: "FunctionProtoTest",
            producer_version: "1.0",
            model_version: 1,
            doc_string: "A test model for model local functions."
          >
         agraph (float[N] x) => (float[N] out)
         {
            out = custom_domain.Selu<gamma=2.0, gamma=3.0>(x)
         }

         <
         domain: "custom_domain",
         opset_import: [ "" : 15],
         doc_string: "Test function proto"
         >
           Selu
           <alpha: float=1.67326319217681884765625, gamma: float=1.05070102214813232421875>
           (X) => (C)
           {
               constant_alpha = Constant<value_float: float=@alpha>()
               constant_gamma = Constant<value_float: float=@gamma>()
               alpha_x = CastLike(constant_alpha, X)
               gamma_x = CastLike(constant_gamma, X)
               exp_x = Exp(X)
               alpha_x_exp_x = Mul(alpha_x, exp_x)
               alpha_x_exp_x_ = Sub(alpha_x_exp_x, alpha_x)
               neg = Mul(gamma_x, alpha_x_exp_x_)
               pos = Mul(gamma_x, X)
               _zero = Constant<value_float=0.0>()
               zero = CastLike(_zero, X)
               less_eq = LessOrEqual(X, zero)
               C = Where(less_eq, neg, pos)
           }
         '''

        model = onnx.parser.parse_model(input)
        checker.check_model(model)


if __name__ == '__main__':
    unittest.main()
