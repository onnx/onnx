# SPDX-License-Identifier: Apache-2.0
import unittest
from typing import Sequence

import onnx
import onnx.checker
import onnx.helper
import onnx.parser
import onnx.shape_inference
from onnx import AttributeProto, TypeProto
from onnx.test.shape_inference_test import TestShapeInferenceHelper


class TestFunctionInference(TestShapeInferenceHelper):
    def _check(
        self,
        function_text: str,
        input_types: Sequence[TypeProto],
        attributes: Sequence[AttributeProto],
        expected_output_types: Sequence[TypeProto],
    ):
        function = onnx.parser.parse_function(function_text)
        result = onnx.checker.infer_check_function(function, input_types, attributes)
        self.assertEqual(len(expected_output_types), len(result))
        for expected, actual in zip(expected_output_types, result):
            self._compare_value_infos(expected, actual)

    def _check_fails(
        self,
        function_text: str,
        input_types: Sequence[TypeProto],
        attributes: Sequence[AttributeProto],
    ):
        function = onnx.parser.parse_function(function_text)

        def invoke_inference():
            onnx.checker.infer_check_function(function, input_types, attributes)

        self.assertRaises(onnx.shape_inference.InferenceError, invoke_inference)

    def test_fi_basic(self):
        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            f (y, z) => (w) {
                x = Add(y, z)
                w = Mul(x, y)
            }
        """
        float_type = onnx.helper.make_tensor_type_proto(1, None)
        int32_type = onnx.helper.make_tensor_type_proto(6, None)
        self._check(code, [float_type, float_type], [], [float_type])
        self._check_fails(code, [float_type, int32_type], [])


if __name__ == "__main__":
    unittest.main()
