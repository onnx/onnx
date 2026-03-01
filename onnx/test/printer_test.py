# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx
from onnx import parser, printer, TensorProto


class TestBasicFunctions(unittest.TestCase):
    def check_graph(self, graph: onnx.GraphProto) -> None:
        self.assertEqual(len(graph.node), 3)
        self.assertEqual(graph.node[0].op_type, "MatMul")
        self.assertEqual(graph.node[1].op_type, "Add")
        self.assertEqual(graph.node[2].op_type, "Softmax")

    def test_parse_graph(self) -> None:
        text0 = """
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul(X, W)
              S = Add(T, B)
              C = Softmax(S)
           }
           """
        graph1 = parser.parse_graph(text0)
        text1 = printer.to_text(graph1)
        graph2 = parser.parse_graph(text1)
        text2 = printer.to_text(graph2)
        # Note that text0 and text1 should be semantically-equivalent, but may differ
        # in white-space and other syntactic sugar. However, we expect text1 and text2
        # to be identical.
        self.assertEqual(text1, text2)
        self.check_graph(graph2)

    def test_low_precision_data_types_printing(self) -> None:
        """Test that low precision data types are printed correctly instead of showing '...'"""
        # Test cases for data types that should use int32_data field
        test_cases = [
            ("float16", TensorProto.FLOAT16, [15360, 16384]),
            ("bfloat16", TensorProto.BFLOAT16, [16256, 16320]),
            ("float8e4m3fn", TensorProto.FLOAT8E4M3FN, [64, 128]),
            ("float8e4m3fnuz", TensorProto.FLOAT8E4M3FNUZ, [64, 128]),
            ("float8e5m2", TensorProto.FLOAT8E5M2, [64, 128]),
            ("float8e5m2fnuz", TensorProto.FLOAT8E5M2FNUZ, [64, 128]),
            ("uint4", TensorProto.UINT4, [0, 15]),
            ("int4", TensorProto.INT4, [0, 8]),  # Using positive values to avoid confusion
            ("float4e2m1", TensorProto.FLOAT4E2M1, [0, 8]),
        ]
        
        for type_name, data_type_enum, test_values in test_cases:
            with self.subTest(data_type=type_name):
                # Create a test model with initializer
                values_str = "{" + ",".join(map(str, test_values)) + "}"
                model_text = f"""
                <
                  ir_version: 10,
                  opset_import: [ "" : 19]
                >
                agraph (float[2] X) => ({type_name}[2] C)
                <
                  {type_name}[{len(test_values)}] weight = {values_str}
                >
                {{
                   C = Cast<to={data_type_enum}>(X)
                }}
                """
                
                # Parse and print the model
                model = parser.parse_model(model_text)
                text_output = printer.to_text(model)
                
                # Verify that the output doesn't contain "..." for the initializer
                self.assertNotIn("...", text_output, 
                               f"Printer should not show '...' for {type_name} data type")
                
                # Verify that the values are preserved in some form
                # (either as the original values or formatted differently)
                values_preserved = any(str(val) in text_output for val in test_values)
                self.assertTrue(values_preserved,
                              f"Values should be preserved in printed output for {type_name}")
                
                # Test round-trip consistency
                model2 = parser.parse_model(text_output)
                text_output2 = printer.to_text(model2)
                self.assertEqual(text_output, text_output2,
                               f"Round-trip should be consistent for {type_name}")


if __name__ == "__main__":
    unittest.main()
