# SPDX-License-Identifier: Apache-2.0
import unittest

import onnx
import onnx.parser
import onnx.shape_inference


class TestModelInference(unittest.TestCase):
    def _check(self, model_text: str, *expected: int):
        """Check that the model inference infers the expected types for outputs.
        Restricted to the simple case of tensor types, so expected types specify
        only the element type (ints corresponding to onnx.TensorProto.DataType).
        """
        model = onnx.parser.parse_model(model_text)
        inferred = onnx.shape_inference.infer_shapes(model)
        outputs = inferred.graph.output
        for (output, expected_elem_type) in zip(outputs, expected):
            inferred_type = output.type
            self.assertTrue(inferred_type.HasField("tensor_type"))
            tensor_type = inferred_type.tensor_type
            self.assertTrue(tensor_type.HasField("elem_type"))
            elem_type = tensor_type.elem_type
            self.assertEqual(elem_type, expected_elem_type)

    def test_mi_basic(self):
        """Test that model inference infers model output type."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17]
            >
            agraph (float[N] x) => (y)
            {
                y = Cast<to=6> (x)
            }
        """
        self._check(model, onnx.TensorProto.INT32)

    def test_mi_function(self):
        """Test use of functions."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            agraph (float[N] x) => (y)
            {
                y = local.cast(x)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            cast (x) => (y)
            {
                y = Cast<to=6> (x)
            }
        """
        self._check(model, onnx.TensorProto.INT32)

    def test_mi_function_attr(self):
        """Test use of functions with attribute parameters."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            agraph (float[N] x) => (y)
            {
                y = local.cast<target=6>(x)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            cast<target>(x) => (y)
            {
                y = Cast<to:int = @target> (x)
            }
        """
        self._check(model, onnx.TensorProto.INT32)

    def test_mi_function_subgraph_attr(self):
        """Test use of function attributes within subgraphs."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            agraph (float[N] x, bool flag) => (y)
            {
                y = local.cast<target=6>(x, flag)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            cast<target>(x, flag) => (y)
            {
                y = If (flag) <
                    then_branch = g1 () => (z_then) { z_then = Cast<to:int = @target> (x) },
                    else_branch = g2 () => (z_else) { z_else = Cast<to:int = @target> (x) }
                    >
            }
        """
        self._check(model, onnx.TensorProto.INT32)

    def test_mi_function_multiple_calls(self):
        """Test use of multiple invocation of functions."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            agraph (float[N] x, bool flag) => (y, z)
            {
                y = local.cast<target=6>(x, flag)
                z = local.cast<target=7>(x, flag)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            cast<target>(x, flag) => (y)
            {
                y = If (flag) <
                    then_branch = g1 () => (z_then) { z_then = Cast<to:int = @target> (x) },
                    else_branch = g2 () => (z_else) { z_else = Cast<to:int = @target> (x) }
                    >
            }
        """
        self._check(model, onnx.TensorProto.INT32, onnx.TensorProto.INT64)


if __name__ == "__main__":
    unittest.main()
