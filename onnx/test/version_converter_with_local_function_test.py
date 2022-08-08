# SPDX-License-Identifier: Apache-2.0

from onnx import checker, parser
import onnx.version_converter
import unittest


class TestVersionConverterLocalFunction(unittest.TestCase):
    def test_local_function(self) -> None:
        input = '''
            <
                ir_version: 8,
                opset_import: [ "" : 2, "custom_domain" : 1],
                producer_name: "FunctionProtoTest",
                producer_version: "1.0",
                model_version: 1,
                doc_string: "A test model for model local functions."
            >
                agraph (float[N] x) => (float[N] out)
                {
                    out = custom_domain.Square(x)
                }
            <
                domain: "custom_domain",
                opset_import: [ "" : 2],
                doc_string: "Test function proto"
            >
            Square(X) => (C)
            {
                C = Mul(X, X)
            }
        '''

        model = onnx.parser.parse_model(input)
        checker.check_model(model, True)
        model_converted = onnx.version_converter.convert_version(model=model, target_version=16)
        checker.check_model(model_converted, True)

    def test_local_function_with_attributes(self) -> None:
        input = '''
            <
            ir_version: 8,
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
           <alpha, gamma>
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
        checker.check_model(model, True)
        model_converted = onnx.version_converter.convert_version(model=model, target_version=16)
        checker.check_model(model_converted, True)


if __name__ == '__main__':
    unittest.main()
