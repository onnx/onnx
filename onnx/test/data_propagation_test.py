# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from shape_inference_test import TestShapeInferenceHelper

import onnx.parser
from onnx import TensorProto
from onnx.helper import make_node, make_tensor, make_tensor_value_info


class TestDataPropagation(TestShapeInferenceHelper):
    def test_expand_symbolic_input(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.INT32, (3, 1, 2)), ("y", TensorProto.INT32, (1, 4, 2))],
            [
                make_node("Shape", ["y"], ["shape"]),
                make_node("Expand", ["x", "shape"], ["z"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.INT32, (3, 4, 2)),
            ],
            data_prop=True,
        )

    def test_constantofshape_with_symbolic_shape(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5))],
            [
                make_node("Shape", ["x"], ["shape"]),
                make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["y"],
                    value=make_tensor("value", TensorProto.INT32, (1,), (2,)),
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, (3,)),
                make_tensor_value_info("y", TensorProto.INT32, (3, 4, 5)),
            ],
            data_prop=True,
        )

    def test_model_data_propagation(self) -> None:
        """Infer the shape of z by propagating the value of xshape."""
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 1, 16] x, float[1, 8, 16] y) => () {
                xshape = Shape (x)
                z = Expand (y, xshape)
            }
        """
        )
        self._assert_inferred(
            model,
            [
                make_tensor_value_info("xshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (4, 8, 16)),
            ],
            data_prop=True,
        )

    def test_data_prop_via_function(self) -> None:
        """Test value-propagation through function calls.
        Underlying core example is same as previous test_model_data_propagation.
        """
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18, "local" : 1 ]>
            agraph (float[4, 1, 16] x, float[1, 8, 16] y) => () {
                xshape = local.GetShape (x)
                z = Expand (y, xshape)
            }
            <domain: "local", opset_import: [ "" : 18 ]>
            GetShape (x) => (shapeval) {
                shapeval = Shape(x)
            }
        """
        )
        self._assert_inferred(
            model,
            [
                make_tensor_value_info("xshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (4, 8, 16)),
            ],
            data_prop=True,
        )

    def test_multiple_calls_to_function(self) -> None:
        """Test value-propagation handles multiple calls to same function correctly.
        Underlying core example is same as previous test_model_data_propagation.
        """
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18, "local" : 1 ]>
            agraph (float[4, 1, 16] x, float[1, 8, 16] y) => () {
                yshape = local.GetShape (y)
                xshape = local.GetShape (x)
                z = Expand (y, xshape)
                w = Expand (y, yshape)
            }
            <domain: "local", opset_import: [ "" : 18 ]>
            GetShape (x) => (shapeval) {
                shapeval = Shape(x)
            }
        """
        )
        self._assert_inferred(
            model,
            [
                make_tensor_value_info("yshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("xshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (4, 8, 16)),
                make_tensor_value_info("w", TensorProto.FLOAT, (1, 8, 16)),
            ],
            data_prop=True,
        )

    def test_shape_arithmetic(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("y", TensorProto.FLOAT, (1, 2, 3))],
            [
                make_node("Shape", ["x"], ["xshape"]),
                make_node("Shape", ["y"], ["yshape"]),
                make_node("Add", ["xshape", "yshape"], ["zshape"]),
                make_node(
                    "ConstantOfShape",
                    ["zshape"],
                    ["z"],
                    value=make_tensor("value", TensorProto.INT32, (1,), (2,)),
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("xshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("yshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("zshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.INT32, (4, 6, 8)),
            ],
            data_prop=True,
        )

    def test_shape_arithmetic_with_broadcast(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("y", TensorProto.FLOAT, (3,))],
            [
                make_node("Shape", ["x"], ["xshape"]),
                make_node("Shape", ["y"], ["yshape"]),
                make_node("Add", ["xshape", "yshape"], ["zshape"]),
                make_node(
                    "ConstantOfShape",
                    ["zshape"],
                    ["z"],
                    value=make_tensor("value", TensorProto.INT32, (1,), (2,)),
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("xshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("yshape", TensorProto.INT64, (1,)),
                make_tensor_value_info("zshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.INT32, (6, 7, 8)),
            ],
            data_prop=True,
        )

    def test_shape_arithmetic_with_zero_broadcast(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ()), ("y", TensorProto.FLOAT, (3,))],
            [
                make_node("Shape", ["x"], ["xshape"]),
                make_node("Shape", ["y"], ["yshape"]),
                make_node("Add", ["xshape", "yshape"], ["zshape"]),
                make_node(
                    "ConstantOfShape",
                    ["zshape"],
                    ["z"],
                    value=make_tensor("value", TensorProto.INT32, (1,), (2,)),
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("xshape", TensorProto.INT64, (0,)),
                make_tensor_value_info("yshape", TensorProto.INT64, (1,)),
                make_tensor_value_info("zshape", TensorProto.INT64, (0,)),
                make_tensor_value_info("z", TensorProto.INT32, ()),
            ],
            data_prop=True,
        )

    def test_empty_tensor(self) -> None:
        """Test that a Concat with an empty tensor as input is handled correctly by data-propagation."""
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[256] y) => (float[N] z)
            <float[0] x = {}>
            {
                z = Concat <axis=0> (x, y)
            }
        """
        )
        inferred_model = onnx.shape_inference.infer_shapes(model, True, True, True)
        output = inferred_model.graph.output[0]
        self.assertEqual(output.type.tensor_type.shape.dim[0].dim_value, 256)

    def test_empty_tensor_negative_axis(self) -> None:
        """Test that a Concat with an empty tensor as input is handled correctly by data-propagation.
        This time with a negative axis.
        """
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[256] y) => (float[N] z)
            <float[0] x = {}>
            {
                z = Concat <axis=-1> (x, y)
            }
        """
        )
        inferred_model = onnx.shape_inference.infer_shapes(model, True, True, True)
        output = inferred_model.graph.output[0]
        self.assertEqual(output.type.tensor_type.shape.dim[0].dim_value, 256)

    def test_symbolic_add_produces_expression(self) -> None:
        """Shape → Add → Reshape: symbolic dim expression propagates to Reshape output."""
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[N, 3, H, W] x, float[M] y) => (float[N, 5, H, W] z) {
                shape = Shape (x)
                delta = Constant <value = int64[4] {0, 2, 0, 0}> ()
                new_shape = Add (shape, delta)
                z = Reshape (y, new_shape)
            }
        """
        )
        inferred_model = onnx.shape_inference.infer_shapes(model, True, True, True)
        output = inferred_model.graph.output[0]
        dims = list(output.type.tensor_type.shape.dim)
        assert len(dims) == 4
        # First dim: N + 0 → should be "N + 0" or similar (symbolic expression)
        assert dims[0].dim_param, f"Expected dim_param, got {dims[0]}"
        assert "N" in dims[0].dim_param
        # Second dim: 3 + 2 = 5 (concrete)
        assert dims[1].dim_value == 5
        # Third dim: H + 0 → should contain "H"
        assert dims[2].dim_param, f"Expected dim_param, got {dims[2]}"
        assert "H" in dims[2].dim_param
        # Fourth dim: W + 0 → should contain "W"
        assert dims[3].dim_param, f"Expected dim_param, got {dims[3]}"
        assert "W" in dims[3].dim_param

    def test_symbolic_sub_produces_expression(self) -> None:
        """Shape → Sub → Reshape: symbolic dim expression propagates to Reshape output."""
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[N, 4, H] x, float[1] y) => (float[N, 3, H] z) {
                shape = Shape (x)
                delta = Constant <value = int64[3] {0, 1, 0}> ()
                new_shape = Sub (shape, delta)
                z = Reshape (y, new_shape)
            }
        """
        )
        inferred_model = onnx.shape_inference.infer_shapes(model, True, True, True)
        output = inferred_model.graph.output[0]
        dims = list(output.type.tensor_type.shape.dim)
        assert len(dims) == 3
        assert dims[0].dim_param, f"Expected dim_param, got {dims[0]}"
        assert "N" in dims[0].dim_param
        assert dims[1].dim_value == 3
        assert dims[2].dim_param, f"Expected dim_param, got {dims[2]}"
        assert "H" in dims[2].dim_param

    def test_symbolic_mul_produces_expression(self) -> None:
        """Shape → Mul → Reshape: symbolic dim expression propagates to Reshape output."""
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[N, 3, H] x, float[1] y) => (float[N, 6, H] z) {
                shape = Shape (x)
                scale = Constant <value = int64[3] {1, 2, 1}> ()
                new_shape = Mul (shape, scale)
                z = Reshape (y, new_shape)
            }
        """
        )
        inferred_model = onnx.shape_inference.infer_shapes(model, True, True, True)
        output = inferred_model.graph.output[0]
        dims = list(output.type.tensor_type.shape.dim)
        assert len(dims) == 3
        assert dims[0].dim_param, f"Expected dim_param, got {dims[0]}"
        assert "N" in dims[0].dim_param
        assert dims[1].dim_value == 6
        assert dims[2].dim_param, f"Expected dim_param, got {dims[2]}"
        assert "H" in dims[2].dim_param

    def test_symbolic_shape_add_reshape(self) -> None:
        """End-to-end: Shape → Add → Reshape with symbolic dims produces expression-based shape."""
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[N, 3] x, float[1] y) => (float[N, 5] z) {
                shape = Shape (x)
                delta = Constant <value = int64[2] {0, 2}> ()
                new_shape = Add (shape, delta)
                z = Reshape (y, new_shape)
            }
        """
        )
        inferred_model = onnx.shape_inference.infer_shapes(model, True, True, True)
        output = inferred_model.graph.output[0]
        dims = list(output.type.tensor_type.shape.dim)
        assert len(dims) == 2
        # First dim should be a symbolic expression containing "N"
        assert dims[0].dim_param, (
            f"Expected dim_param, got dim_value={dims[0].dim_value}"
        )
        assert "N" in dims[0].dim_param
        # Second dim should be concrete 5
        assert dims[1].dim_value == 5

    def test_symbolic_shape_concat_reshape(self) -> None:
        """End-to-end: Shape → Reshape with symbolic + concrete dims."""
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[N, 3, H, W] x, float[1] y) => (float[N, 3, H, W] z) {
                shape = Shape (x)
                z = Reshape (y, shape)
            }
        """
        )
        inferred_model = onnx.shape_inference.infer_shapes(model, True, True, True)
        output = inferred_model.graph.output[0]
        dims = list(output.type.tensor_type.shape.dim)
        assert len(dims) == 4
        assert dims[0].dim_param == "N"
        assert dims[1].dim_value == 3
        assert dims[2].dim_param == "H"
        assert dims[3].dim_param == "W"


if __name__ == "__main__":
    unittest.main()
