# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx import OperatorSetIdProto, reference
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect


class SeluFunction(Base):
    @staticmethod
    def export() -> None:
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

        # save model and test data for runtime test
        sess = reference.ReferenceEvaluator(model)
        x = np.random.randn(3).astype(np.float32)
        input_dict = {"x": x}
        y = sess.run(None, input_dict)

        expect(model, inputs=[x], outputs=[y[0]], name="test_selu_function")


    @staticmethod
    def export() -> None:
        default_alpha = 1.67326319217681884765625
        default_gamma = 1.05070102214813232421875
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

        graph_texts = [
            (
                "agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu(x) }",
                "all_attribute_default",
            ),
            (
                "agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<alpha=2.0>(x) }",
                "gamma_attribute_default",
            ),
            (
                "agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<gamma=3.0>(x) }",
                "alpha_attribute_default",
            ),
            (
                "agraph (float[N] x) => (float[N] out) { out = custom_domain.Selu<alpha=2.0, gamma=3.0>(x) }",
                "no_attribute_default",
            ),
        ]

        for graph_text, test_name_suffix in graph_texts:
            graph = onnx.parser.parse_graph(graph_text)
            opset_imports = [
                OperatorSetIdProto(domain="", version=15),
                OperatorSetIdProto(domain="custom_domain", version=1),
            ]

            model = onnx.helper.make_model(
                graph, functions=functions, opset_imports=opset_imports
            )

            # save model and test data for runtime test
            sess = reference.ReferenceEvaluator(model)
            x = np.random.randn(3).astype(np.float32)
            input_dict = {"x": x}
            y = sess.run(None, input_dict)

            expect(
                model,
                inputs=[x],
                outputs=[y[0]],
                name=f"test_selu_function_{test_name_suffix}",
            )
