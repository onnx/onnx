# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from shape_inference_test_utils import TestShapeInferenceHelper

import onnx.parser
from onnx import TensorProto
from onnx.helper import make_node, make_tensor, make_tensor_value_info


class TestDataPropagation(TestShapeInferenceHelper):
    @pytest.mark.parametrize("version", [13, 19, 21, 23, 24, 25])
    def test_size_of_float_tensor_uses_shape(self, version: int) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3))],
            [
                make_node("Size", ["x"], ["size"]),
                make_node("Unsqueeze", ["size", "axis"], ["shape"]),
                make_node("ConstantOfShape", ["shape"], ["y"]),
            ],
            [],
            initializer=[make_tensor("axis", TensorProto.INT64, (1,), [0])],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("size", TensorProto.INT64, ()),
                make_tensor_value_info("shape", TensorProto.INT64, (1,)),
                make_tensor_value_info("y", TensorProto.FLOAT, (6,)),
            ],
            data_prop=True,
            opset_imports=[onnx.helper.make_opsetid("", version)],
        )

    def test_symbolic_int32_overflow_does_not_alias_dimension_arithmetic(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ("N",)), ("matrix", TensorProto.FLOAT, ("N", 2))],
            [
                make_node("Shape", ["x"], ["shape64"]),
                make_node("Cast", ["shape64"], ["shape32"], to=TensorProto.INT32),
                make_node("Mul", ["shape32", "two"], ["wrapped"]),
                make_node("Cast", ["wrapped"], ["shape"], to=TensorProto.INT64),
                make_node("ConstantOfShape", ["shape"], ["y"]),
                make_node("Flatten", ["matrix"], ["flat"], axis=0),
            ],
            [],
            initializer=[make_tensor("two", TensorProto.INT32, (1,), [2])],
        )
        inferred = self._inferred(graph, data_prop=True)
        shapes = {
            info.name: info.type.tensor_type.shape for info in inferred.graph.value_info
        }
        assert shapes["flat"].dim[1].dim_param == "N*2"
        assert shapes["y"].dim[0].dim_param != "N*2"

    @pytest.mark.parametrize(
        ("operation", "spelling"),
        [("Add", "N + 2"), ("Sub", "N - 2"), ("Mul", "N*2"), ("Div", "N/2")],
    )
    @pytest.mark.parametrize("reverse", [False, True])
    def test_expression_names_share_shape_and_data_provenance(
        self, operation: str, spelling: str, reverse: bool
    ) -> None:
        nodes = [
            make_node("Shape", ["x"], ["shape"]),
            make_node(operation, ["shape", "two"], ["computed"]),
            make_node("ConstantOfShape", ["computed"], ["y"]),
            make_node(operation, ["shape", "two"], ["again"]),
            make_node("ConstantOfShape", ["again"], ["repeated"]),
        ]
        flat = make_node("Flatten", ["matrix"], ["flat"], axis=0)
        if reverse:
            nodes.insert(0, flat)
        else:
            nodes.append(flat)
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("N",)),
                ("matrix", TensorProto.FLOAT, ("N", 2)),
                ("independent", TensorProto.FLOAT, (spelling,)),
            ],
            nodes,
            [],
            initializer=[make_tensor("two", TensorProto.INT64, (1,), [2])],
        )
        inferred = self._inferred(graph, data_prop=True)
        shapes = {
            info.name: info.type.tensor_type.shape for info in inferred.graph.value_info
        }
        name = shapes["y"].dim[0].dim_param
        assert name and name != spelling
        assert shapes["repeated"] == shapes["y"]
        if operation == "Mul":
            assert shapes["flat"].dim[1].dim_param == name

    @pytest.mark.parametrize("op_type", ["Identity", "Neg"])
    def test_large_nonconstant_integer_vector(self, op_type: str) -> None:
        graph = self._make_graph(
            [("x", TensorProto.INT64, (1 << 40,))],
            [make_node(op_type, ["x"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (1 << 40,))],
            data_prop=True,
        )

    def test_inconsistent_partial_value_count_is_discarded(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,))],
            [
                make_node("Unsqueeze", ["pads", "axis"], ["matrix"]),
                make_node("Gather", ["matrix", "index"], ["selected"]),
                make_node("Add", ["selected", "zeros"], ["computed_pads"]),
                make_node("Pad", ["x", "computed_pads"], ["y"]),
            ],
            [],
            initializer=[
                make_tensor("pads", TensorProto.INT64, (2,), [1, 2]),
                make_tensor("axis", TensorProto.INT64, (1,), [0]),
                make_tensor("index", TensorProto.INT64, (), [0]),
                make_tensor("zeros", TensorProto.INT64, (2,), [0, 0]),
            ],
        )
        inferred = self._inferred(graph, data_prop=True)
        y = next(info for info in inferred.graph.value_info if info.name == "y")
        dim = y.type.tensor_type.shape.dim[0]
        assert not dim.HasField("dim_value") or dim.dim_value == 6

    @pytest.mark.parametrize("version", [13, 27])
    def test_if_branch_data_is_local(self, version: int) -> None:
        model = onnx.parser.parse_model("""
            <ir_version: 8, opset_import: ["" : 13]>
            test (bool cond, float[N, 3] x) => (y) {
                outer_shape = Shape(x)
                y = If(cond) <
                    then_branch = then_graph () => (branch_output) {
                        local = Constant<value = int64[1] {2}>()
                        shared = Identity(local)
                        branch_output = ConstantOfShape(shared)
                    },
                    else_branch = else_graph () => (branch_output) {
                        local = Constant<value = int64[1] {4}>()
                        shared = Identity(local)
                        branch_output = ConstantOfShape(shared)
                    }
                >
                z = If(cond) <
                    then_branch = outer_then () => (branch_output) {
                        branch_shape = Identity(outer_shape)
                        branch_output = ConstantOfShape(branch_shape)
                    },
                    else_branch = outer_else () => (branch_output) {
                        branch_shape = Identity(outer_shape)
                        branch_output = ConstantOfShape(branch_shape)
                    }
                >
            }
        """)
        model.opset_import[0].version = version
        inferred = self._inferred(model, data_prop=True)
        assert (
            not inferred.graph.output[0]
            .type.tensor_type.shape.dim[0]
            .HasField("dim_value")
        )
        z = next(info for info in inferred.graph.value_info if info.name == "z")
        assert z.type.tensor_type.shape.dim[0].dim_param == "N"
        assert z.type.tensor_type.shape.dim[1].dim_value == 3

    @pytest.mark.parametrize("operation", ["Squeeze", "Gather"])
    def test_partial_values_do_not_create_malformed_pad_data(
        self, operation: str
    ) -> None:
        nodes = [
            make_node("Unsqueeze", ["pads", "axis"], ["matrix"]),
            make_node(
                operation,
                ["matrix", "runtime"]
                if operation == "Squeeze"
                else ["matrix", "index"],
                ["propagated"],
            ),
            make_node("Pad", ["x", "propagated"], ["y"]),
        ]
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,)), ("runtime", TensorProto.INT64, (1,))],
            nodes,
            [],
            initializer=[
                make_tensor("pads", TensorProto.INT64, (2,), [1, 2]),
                make_tensor("axis", TensorProto.INT64, (1,), [0]),
                make_tensor("index", TensorProto.INT64, (), [0]),
            ],
        )
        self._inferred(graph, data_prop=True)

    @pytest.mark.parametrize(
        ("op_type", "versions"), [("Neg", [6, 13]), ("Div", [7, 13, 14])]
    )
    def test_integer_unary_division_overflow(
        self, op_type: str, versions: list[int]
    ) -> None:
        for version in versions:
            graph = self._make_graph(
                [],
                [
                    make_node(
                        op_type,
                        ["x"] if op_type == "Neg" else ["x", "minus_one"],
                        ["y"],
                    )
                ],
                [],
                initializer=[
                    make_tensor("x", TensorProto.INT64, (1,), [-(1 << 63)]),
                    make_tensor("minus_one", TensorProto.INT64, (1,), [-1]),
                ],
            )
            with pytest.raises(onnx.shape_inference.InferenceError, match="overflow"):
                self._inferred(
                    graph,
                    data_prop=True,
                    opset_imports=[onnx.helper.make_opsetid("", version)],
                )

    @pytest.mark.parametrize("version", [13, 14, 27])
    def test_float_division_does_not_propagate_integer_quotient(
        self, version: int
    ) -> None:
        graph = self._make_graph(
            [],
            [
                make_node("Cast", ["a"], ["af"], to=TensorProto.FLOAT),
                make_node("Cast", ["b"], ["bf"], to=TensorProto.FLOAT),
                make_node("Div", ["af", "bf"], ["quotient"]),
                make_node("Ceil", ["quotient"], ["rounded"]),
                make_node("Cast", ["rounded"], ["shape"], to=TensorProto.INT64),
                make_node("ConstantOfShape", ["shape"], ["y"]),
            ],
            [],
            initializer=[
                make_tensor("a", TensorProto.INT64, (1,), [5]),
                make_tensor("b", TensorProto.INT64, (1,), [2]),
            ],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("af", TensorProto.FLOAT, (1,)),
                make_tensor_value_info("bf", TensorProto.FLOAT, (1,)),
                make_tensor_value_info("quotient", TensorProto.FLOAT, (1,)),
                make_tensor_value_info("rounded", TensorProto.FLOAT, (1,)),
                make_tensor_value_info("shape", TensorProto.INT64, (1,)),
                make_tensor_value_info("y", TensorProto.FLOAT, (None,)),
            ],
            data_prop=True,
            opset_imports=[onnx.helper.make_opsetid("", version)],
        )

    def test_narrowing_cast_discards_out_of_range_data(self) -> None:
        graph = self._make_graph(
            [],
            [
                make_node("Cast", ["x"], ["narrowed"], to=TensorProto.INT32),
                make_node("Cast", ["narrowed"], ["shape"], to=TensorProto.INT64),
                make_node("Relu", ["shape"], ["positive_shape"]),
                make_node("ConstantOfShape", ["positive_shape"], ["y"]),
            ],
            [],
            initializer=[make_tensor("x", TensorProto.INT64, (1,), [1 << 31])],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("narrowed", TensorProto.INT32, (1,)),
                make_tensor_value_info("shape", TensorProto.INT64, (1,)),
                make_tensor_value_info("positive_shape", TensorProto.INT64, (1,)),
                make_tensor_value_info("y", TensorProto.FLOAT, (None,)),
            ],
            data_prop=True,
        )

    @pytest.mark.parametrize("op_type", ["Add", "Mul"])
    def test_symbolic_expression_growth_is_bounded(self, op_type: str) -> None:
        nodes = [make_node("Shape", ["x"], ["value_0"])]
        nodes.extend(
            make_node(op_type, [f"value_{i}", f"value_{i}"], [f"value_{i + 1}"])
            for i in range(16)
        )
        nodes.append(make_node("ConstantOfShape", ["value_16"], ["y"]))
        graph = self._make_graph([("x", TensorProto.FLOAT, ("N",))], nodes, [])
        inferred = self._inferred(graph, data_prop=True)
        y = next(info for info in inferred.graph.value_info if info.name == "y")
        dim = y.type.tensor_type.shape.dim[0]
        assert not dim.HasField("dim_value")
        assert len(dim.dim_param) <= 4096

    @pytest.mark.parametrize(
        ("start", "limit", "delta", "index", "expected_dim"),
        [
            (0, (1 << 53) + 1, 1 << 53, 1, 1 << 53),
            (-(1 << 63), (1 << 63) - 1, (1 << 63) - 1, 2, (1 << 63) - 2),
            ((1 << 63) - 1, -(1 << 63), -((1 << 63) - 1), 1, 0),
            ((1 << 63) - 1, -(1 << 63), -(1 << 63), 0, (1 << 63) - 1),
        ],
    )
    def test_range_exact_int64_data(
        self, start: int, limit: int, delta: int, index: int, expected_dim: int
    ) -> None:
        graph = self._make_graph(
            [],
            [
                make_node("Range", ["start", "limit", "delta"], ["values"]),
                make_node("Gather", ["values", "index"], ["shape"]),
                make_node("ConstantOfShape", ["shape"], ["y"]),
            ],
            [],
            initializer=[
                make_tensor("start", TensorProto.INT64, (), [start]),
                make_tensor("limit", TensorProto.INT64, (), [limit]),
                make_tensor("delta", TensorProto.INT64, (), [delta]),
                make_tensor("index", TensorProto.INT64, (1,), [index]),
            ],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "values", TensorProto.INT64, (len(range(start, limit, delta)),)
                ),
                make_tensor_value_info("shape", TensorProto.INT64, (1,)),
                make_tensor_value_info("y", TensorProto.FLOAT, (expected_dim,)),
            ],
            data_prop=True,
        )

    @pytest.mark.parametrize("op_type", ["Add", "Sub", "Mul", "Div"])
    @pytest.mark.parametrize("version", [13, 18])
    def test_arithmetic_multidimensional_broadcast(
        self, op_type: str, version: int
    ) -> None:
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph () => (z)
            <int64 start = {1}, int64 row_limit = {3}, int64 col_limit = {4},
             int64 delta = {1}, int64[1] axis0 = {0}, int64[1] axis1 = {1}>
            {
                row_values = Range (start, row_limit, delta)
                col_values = Range (start, col_limit, delta)
                rows = Unsqueeze (row_values, axis1)
                cols = Unsqueeze (col_values, axis0)
                z = BINARY_OP (rows, cols)
            }
            """.replace("BINARY_OP", op_type)
        )
        model.opset_import[0].version = version
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        dims = inferred.graph.output[0].type.tensor_type.shape.dim
        assert [dim.dim_value for dim in dims] == [2, 3]

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
            agraph (float[N, 4, H] x, float[M] y) => (float[N, 3, H] z) {
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
