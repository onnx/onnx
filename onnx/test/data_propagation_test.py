# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
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

    def test_add_overflow(self) -> None:
        # Add with INT64_MAX + 1 must raise InferenceError rather than silently wrapping.
        # Uses Shape to inject INT64_MAX as a known int64 value, then adds 1 via Shape.
        INT64_MAX = (1 << 63) - 1
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (INT64_MAX,)),
                ("y", TensorProto.FLOAT, (1,)),
            ],
            [
                make_node("Shape", ["x"], ["xshape"]),
                make_node("Shape", ["y"], ["yshape"]),
                make_node("Add", ["xshape", "yshape"], ["zshape"]),
                make_node(
                    "ConstantOfShape",
                    ["zshape"],
                    ["z"],
                    value=make_tensor("value", TensorProto.INT32, (1,), (0,)),
                ),
            ],
            [],
        )
        with pytest.raises(onnx.shape_inference.InferenceError):
            self._assert_inferred(graph, [], data_prop=True)

    def test_sub_overflow(self) -> None:
        # Sub with INT64_MIN - 1 must raise InferenceError rather than silently wrapping.
        # Uses Constant nodes to inject the boundary values directly.
        INT64_MIN = -(1 << 63)
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["a"],
                    value=make_tensor("a", TensorProto.INT64, (1,), [INT64_MIN]),
                ),
                make_node(
                    "Constant",
                    [],
                    ["b"],
                    value=make_tensor("b", TensorProto.INT64, (1,), [1]),
                ),
                make_node("Sub", ["a", "b"], ["zshape"]),
                make_node(
                    "ConstantOfShape",
                    ["zshape"],
                    ["z"],
                    value=make_tensor("value", TensorProto.INT32, (1,), (0,)),
                ),
            ],
            [],
        )
        with pytest.raises(onnx.shape_inference.InferenceError):
            self._assert_inferred(graph, [], data_prop=True)

    def test_mul_overflow(self) -> None:
        # Mul with INT64_MAX * 2 must raise InferenceError rather than silently wrapping.
        INT64_MAX = (1 << 63) - 1
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["a"],
                    value=make_tensor("a", TensorProto.INT64, (1,), [INT64_MAX]),
                ),
                make_node(
                    "Constant",
                    [],
                    ["b"],
                    value=make_tensor("b", TensorProto.INT64, (1,), [2]),
                ),
                make_node("Mul", ["a", "b"], ["zshape"]),
                make_node(
                    "ConstantOfShape",
                    ["zshape"],
                    ["z"],
                    value=make_tensor("value", TensorProto.INT32, (1,), (0,)),
                ),
            ],
            [],
        )
        with pytest.raises(onnx.shape_inference.InferenceError):
            self._assert_inferred(graph, [], data_prop=True)

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
        assert output.type.tensor_type.shape.dim[0].dim_value == 256

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
        assert output.type.tensor_type.shape.dim[0].dim_value == 256
