# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
import unittest

from parameterized import parameterized

import onnx
from onnx import GraphProto, OperatorSetIdProto, checker


class TestBasicFunctions(unittest.TestCase):
    def check_graph(self, graph: GraphProto) -> None:
        self.assertEqual(len(graph.node), 3)
        self.assertEqual(graph.node[0].op_type, "MatMul")
        self.assertEqual(graph.node[1].op_type, "Add")
        self.assertEqual(graph.node[2].op_type, "Softmax")

    def test_parse_graph(self) -> None:
        input = """
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul(X, W)
              S = Add(T, B)
              C = Softmax(S)
           }
           """
        graph = onnx.parser.parse_graph(input)
        self.check_graph(graph)

    def test_parse_model(self) -> None:
        input = """
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
           """
        model = onnx.parser.parse_model(input)
        self.assertEqual(model.ir_version, 7)
        self.assertEqual(len(model.opset_import), 2)
        self.check_graph(model.graph)

    def test_parse_graph_error(self) -> None:
        input = """
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul[X, W]
              S = Add(T, B)
              C = Softmax(S)
           }
           """
        self.assertRaises(
            onnx.parser.ParseError, lambda: onnx.parser.parse_graph(input)
        )

    def test_parse_model_error(self) -> None:
        input = """
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
           """
        self.assertRaises(
            onnx.parser.ParseError, lambda: onnx.parser.parse_model(input)
        )

    def test_parse_function_with_attributes(self) -> None:
        input = """
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
            out = custom_domain.Selu<alpha=2.0, gamma=3.0>(x)
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
        """

        model = onnx.parser.parse_model(input)
        checker.check_model(model)

    def test_parse_afline_grid(self) -> None:
        input = """
            <
            ir_version: 8,
            opset_import: [ "" : 18, "custom_domain" : 1],
            producer_name: "FunctionProtoTest",
            producer_version: "1.0",
            model_version: 1,
            doc_string: "A test model for model local functions."
          >
         agraph (float[N, DIM, DIM_PLUS_1] theta, int64[K] size) => (float[N, C, H, W, DIM] grid)
         {
            grid = custom_domain.AffineGrid<align_corners=0>(theta, size)
         }

         <
         domain: "custom_domain",
         opset_import: [ "" : 18],
         doc_string: "Test function proto"
         >
           AffineGrid
           <align_corners: int=0>
           (theta, size) => (grid)
           {
                int_zero = Constant<value_int: int=0>()
                int_four = Constant<value_int: int=4>()

                constant_align_corners = Constant<value_int: int=@align_corners>()
                constant_align_corners_equal_zero = Equal (constant_align_corners, int_zero) 
                
                size_ndim = Size (size)
                condition_is_2d = Equal (size_ndim, int_four)

                grid = If (condition_is_2d) <
                    then_branch = g1 () => (float[N, H, W, 2] grid_2d_then) {
                        minus_one = Constant <value = float {-1.0}>()
                        zero = Constant <value = float {0.0}>()
                        one = Constant <value = float {1.0}>()
                        two = Constant <value = float {2.0}>()
                        N, C, H, W = Split <num_outputs: int=4>(size)
                        int_two_1d = Constant<value_ints=[2]>()
                        int_four_1d = Constant<value_ints=[4]>()
                        constant_H_W_shape = Slice(size, int_two_1d, int_four_1d) # [N, C, H, W] => [H, W]
                        zero_H_by_W = ConstantOfShape (constant_H_W_shape)
                        ones_H_by_W = Add (zero_H_by_W, one)

                        H_float = CastLike (H, zero)
                        W_float = CastLike (W, zero)
                        grid_h_0, grid_w_0 = If (constant_align_corners_equal_zero) <
                            then_branch = h1 () => (float[H] grid_h_then, float[W] grid_w_then) {
                                step_h = Div (two, H_float)
                                step_w = Div (two, W_float)
                                step_h_half = Div (step_h, two)
                                start_h = Add (minus_one, step_h_half)
                                grid_h_then = Range (start_h, one, step_h)

                                step_w_half = Div (step_w, two)
                                start_w = Add (minus_one, step_w_half)
                                grid_w_then = Range (start_w, one, step_w)
                            },
                            else_branch = h2 () => (float[H] grid_h_else, float[W] grid_w_else) {
                                H_float_nimus_one = Sub (H_float, one)
                                W_float_nimus_one = Sub (W_float, one)
                                step_h = Div (two, H_float_nimus_one)
                                step_w = Div (two, W_float_nimus_one)
                                epsilon = Constant <value = float {1e-6}>()
                                one_plus_epsilon = Add (one, epsilon)
                                grid_h_else = Range (minus_one, one_plus_epsilon, step_h)
                                grid_w_else = Range (minus_one, one_plus_epsilon, step_w)
                            }
                        >
                        size_ones_H_by_W_transpose = Transpose (ones_H_by_W) # (3, 2)
                        grid_h_1 = Mul(size_ones_H_by_W_transpose, grid_h_0)
                        grid_h = Transpose (grid_h_1)
                        
                        grid_w = Add (grid_w_0, zero_H_by_W)

                        # make folowing a function (theta, grid_w, grid_h) =>  (grid)
                        original_grid_seq = SequenceConstruct (grid_w, grid_h, ones_H_by_W)   
                        original_grid = ConcatFromSequence <axis: int=-1, new_axis: int=1> (original_grid_seq)
                        constant_shape_HW_3 = Constant <value_ints: ints = [-1, 3]> ()
                        original_grid_HW_3 = Reshape (original_grid, constant_shape_HW_3)
                        original_grid_3_HW = Transpose (original_grid_HW_3)

                        grid_N_2_HW = MatMul (theta, original_grid_3_HW)
                        grid_N_HW_2 = Transpose <perm = [0, 2, 1]> (grid_N_2_HW)
                        N_H_W_2_seq = SequenceConstruct (N, H, W, int_two_1d)
                        N_H_W_2 = ConcatFromSequence <axis: int=-1, new_axis: int=0> (N_H_W_2_seq)
                        grid_2d_then = Reshape(grid_N_HW_2, N_H_W_2)
                        },
                    else_branch = g2 () => (float[N, D, H, W, 3] grid_3d_else) {
                        minus_one = Constant <value = float {-1.0}>()
                        zero = Constant <value = float {0.0}>()
                        one = Constant <value = float {1.0}>()
                        two = Constant <value = float {2.0}>()
                        N, C, D, H, W = Split <num_outputs: int=5>(size)
                        int_two_1d = Constant<value_ints=[2]>()
                        int_three_1d = Constant<value_ints=[3]>()
                        int_five_1d = Constant<value_ints=[5]>()
                        constant_D_H_W_shape = Slice(size, int_two_1d, int_five_1d) # [N, C, D, H, W] => [D, H, W]
                        zero_D_H_W = ConstantOfShape (constant_D_H_W_shape)
                        ones_D_H_W = Add (zero_D_H_W, one)

                        D_float = CastLike (D, zero)
                        H_float = CastLike (H, zero)
                        W_float = CastLike (W, zero)
                        grid_d_0, grid_h_0, grid_w_0 = If (constant_align_corners_equal_zero) <
                            then_branch = h1 () => (float[D] grid_d_then, float[H] grid_h_then, float[W] grid_w_then) {
                                step_d = Div (two, D_float)
                                step_h = Div (two, H_float)
                                step_w = Div (two, W_float)

                                step_d_half = Div (step_d, two)
                                start_d = Add (minus_one, step_d_half)
                                grid_d_then = Range (start_d, one, step_d)

                                step_h_half = Div (step_h, two)
                                start_h = Add (minus_one, step_h_half)
                                grid_h_then = Range (start_h, one, step_h)

                                step_w_half = Div (step_w, two)
                                start_w = Add (minus_one, step_w_half)
                                grid_w_then = Range (start_w, one, step_w)
                            },
                            else_branch = h2 () => (float[D] grid_d_else, float[H] grid_h_else, float[W] grid_w_else) {
                                D_float_nimus_one = Sub (D_float, one)
                                H_float_nimus_one = Sub (H_float, one)
                                W_float_nimus_one = Sub (W_float, one)
                                step_d = Div (two, D_float_nimus_one)
                                step_h = Div (two, H_float_nimus_one)
                                step_w = Div (two, W_float_nimus_one)
                                epsilon = Constant <value = float {1e-6}>()
                                one_plus_epsilon = Add (one, epsilon)
                                grid_d_else = Range (minus_one, one_plus_epsilon, step_d)
                                grid_h_else = Range (minus_one, one_plus_epsilon, step_h)
                                grid_w_else = Range (minus_one, one_plus_epsilon, step_w)
                            }
                        >
                        ones_H_W_D = Transpose <perm = [1, 2, 0]> (ones_D_H_W)
                        grid_d_1 = Mul(ones_H_W_D, grid_d_0)
                        grid_d = Transpose <perm = [2, 0, 1]>(grid_d_1)

                        ones_D_W_H = Transpose <perm = [0, 2, 1]> (ones_D_H_W)
                        grid_h_1 = Mul(ones_D_W_H, grid_h_0)
                        grid_h = Transpose <perm = [0, 2, 1]>(grid_h_1)

                        grid_w = Add (grid_w_0, zero_D_H_W)

                        original_grid_seq = SequenceConstruct (grid_w, grid_h, grid_d, ones_D_H_W)
                        original_grid = ConcatFromSequence <axis: int=-1, new_axis: int=1> (original_grid_seq)
                        constant_shape_DHW_4 = Constant <value_ints: ints = [-1, 4]> ()
                        original_grid_DHW_4 = Reshape (original_grid, constant_shape_DHW_4)
                        original_grid_4_DHW = Transpose (original_grid_DHW_4)

                        grid_N_3_DHW = MatMul (theta, original_grid_4_DHW)
                        grid_N_DHW_3 = Transpose <perm = [0, 2, 1]> (grid_N_3_DHW)
                        N_D_H_W_3_seq = SequenceConstruct (N, D, H, W, int_three_1d)
                        N_D_H_W_3 = ConcatFromSequence <axis: int=-1, new_axis: int=0> (N_D_H_W_3_seq)
                        grid_3d_else = Reshape(grid_N_DHW_3, N_D_H_W_3)
                        }
                    >
           }
        """

        model = onnx.parser.parse_model(input)
        checker.check_model(model)
        onnx.save(model, "C:/Temp/affine_grid_test.onnx")

        import torch
        from torch.nn.functional import affine_grid
        import numpy as np
        from onnxruntime import InferenceSession
        inference_session = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        np.random.seed(42)

        test_2d = True
        align_corners = False
        if test_2d:
            N, C, H, W = 32, 3, 240, 512
            theta = np.random.randn(N, 2, 3).astype(np.float32)
            size = np.array([N, C, H, W], dtype=np.int64)
            res = inference_session.run(None, {"theta": theta, "size": size})
            print(res[0])

            t_res = affine_grid(torch.from_numpy(theta), torch.Size((N, C, H, W)), align_corners=align_corners)
            print(t_res)
            np.testing.assert_allclose(res[0], t_res.numpy(), rtol=1e-04, atol=1e-04)
        else:
            N, C, D, H, W = 16, 3, 100, 300, 406
            theta = np.random.randn(N, 3, 4).astype(np.float32)
            size = np.array([N, C, D, H, W], dtype=np.int64)
            res = inference_session.run(None, {"theta": theta, "size": size})
            print(res[0])

            t_res = affine_grid(torch.from_numpy(theta), torch.Size((N, C, D, H, W)), align_corners=align_corners)
            print(t_res)
            np.testing.assert_allclose(res[0], t_res.numpy(), rtol=1e-04, atol=1e-04)


    @parameterized.expand(
        [
            (
                "agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu(x) }",
                {},
            ),
            (
                "agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<alpha=2.0>(x) }",
                {"alpha": 2.0},
            ),
            (
                "agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<gamma=3.0>(x) }",
                {"gamma": 3.0},
            ),
            (
                "agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<alpha=2.0, gamma=3.0>(x) }",
                {"alpha": 2.0, "gamma": 3.0},
            ),
        ]
    )
    def test_composite_parse_function_with_attributes(
        self, graph_text: str, expected_attribute: dict
    ) -> None:
        default_alpha = 1.67326319217681884765625
        default_gamma = 1.05070102214813232421875

        def expect_custom_node_attribute(node, attributes):
            for key in attributes:
                match_attr = [attr for attr in node.attribute if attr.name == key]
                assert len(match_attr) == 1
                assert match_attr[0].f == attributes[key]

        def expect_model_function_attribute(model):
            assert len(model.functions[0].attribute_proto) == 2
            attr_proto_alpha = [
                attr_proto
                for attr_proto in model.functions[0].attribute_proto
                if attr_proto.name == "alpha"
            ]
            assert len(attr_proto_alpha) == 1 and attr_proto_alpha[0].f == default_alpha
            attr_proto_gamma = [
                attr_proto
                for attr_proto in model.functions[0].attribute_proto
                if attr_proto.name == "gamma"
            ]
            assert len(attr_proto_gamma) == 1 and attr_proto_gamma[0].f == default_gamma

        function_text = f"""
         <
         domain: "custom_domain",
         opset_import: [ "" : 15],
         doc_string: "Test function proto"
         >
           Selu
           <alpha: float={default_alpha}, gamma: float={default_gamma}>
           (X) => (C)
           {{
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
           }}
        """

        functions = [onnx.parser.parse_function(function_text)]
        graph = onnx.parser.parse_graph(graph_text)
        opset_imports = [
            OperatorSetIdProto(domain="", version=15),
            OperatorSetIdProto(domain="custom_domain", version=1),
        ]

        model = onnx.helper.make_model(
            graph, functions=functions, opset_imports=opset_imports
        )
        checker.check_model(model)

        expect_model_function_attribute(model)
        expect_custom_node_attribute(model.graph.node[0], expected_attribute)

    def test_parse_node(self):
        node = onnx.parser.parse_node(
            "out1, out2 = SomeDomain.SomeOp <attr1 = 1> (in1, in2)"
        )
        self.assertEqual(list(node.input), ["in1", "in2"])
        self.assertEqual(list(node.output), ["out1", "out2"])
        self.assertEqual(len(node.attribute), 1)
        attr_val = onnx.helper.get_node_attr_value(node, "attr1")
        self.assertEqual(attr_val, 1)
        self.assertEqual(node.domain, "SomeDomain")
        self.assertEqual(node.op_type, "SomeOp")


if __name__ == "__main__":
    unittest.main()
