# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import (
    AttributeProto,
    TensorProto,
    checker,
    helper,
    inliner,
    numpy_helper,
    parser,
)
from onnx.reference import ReferenceEvaluator


class TestInliner:
    def test_basic(self):
        model = parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = local.bar(temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul (x, x)
            }
        """
        )
        inlined = inliner.inline_local_functions(model)
        inlined_nodes = inlined.graph.node
        # function-call should be replaced by Add, followed by Mul
        assert len(inlined_nodes) == 2
        assert inlined_nodes[0].op_type == "Add"
        assert inlined_nodes[1].op_type == "Mul"

    def test_selective_inlining(self):
        model = parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.square (X)
                Y = local.double_and_square (T)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            double_and_square (x) => (y) {
                double = Add(x, x)
                y = local.square(double)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            square (x) => (y) {
                y = Mul (x, x)
            }
        """
        )
        inlined = inliner.inline_selected_functions(
            model, [("local", "square")], exclude=False
        )
        inlined_nodes = inlined.graph.node
        # function-call to square should be replaced by Add, but not the one to double_and_square
        assert len(inlined_nodes) == 2
        assert inlined_nodes[0].op_type == "Mul"
        assert inlined_nodes[1].op_type == "double_and_square"

        # check call to square inside double_and_square was inlined:
        function_nodes = inlined.functions[0].node
        assert len(function_nodes) == 2
        assert function_nodes[0].op_type == "Add"
        assert function_nodes[1].op_type == "Mul"

    def test_selective_exclusion(self):
        model = parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                T = local.square (X)
                Y = local.double_and_square (T)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            double_and_square (x) => (y) {
                double = Add(x, x)
                y = local.square(double)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            square (x) => (y) {
                y = Mul (x, x)
            }
        """
        )
        inlined = inliner.inline_selected_functions(
            model, [("local", "double_and_square")], exclude=True
        )
        inlined_nodes = inlined.graph.node
        # function-call to square should be replaced by Add, but not the one to double_and_square
        assert len(inlined_nodes) == 2
        assert inlined_nodes[0].op_type == "Mul"
        assert inlined_nodes[1].op_type == "double_and_square"

        # check call to square inside double_and_square was inlined:
        function_nodes = inlined.functions[0].node
        assert len(function_nodes) == 2
        assert function_nodes[0].op_type == "Add"
        assert function_nodes[1].op_type == "Mul"

    def test_inline_rejects_cyclic_function(self):
        model = parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y) { Y = local.foo (X) }
            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) { y = local.foo (x) }
        """
        )
        with pytest.raises(checker.ValidationError):
            inliner.inline_local_functions(model)

    def test_schema_function_inlining(self):
        model = parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 20]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = Softsign (X)
            }
        """
        )
        inlined = inliner.inline_selected_functions(
            model, [], exclude=True, inline_schema_functions=True
        )
        inlined_nodes = inlined.graph.node
        assert "Abs" in [n.op_type for n in inlined_nodes]

    def test_sequence_map_inlining_rejects_missing_input_type(self):
        model = parser.parse_model(
            """
            <ir_version: 13, opset_import: [ "" : 17 ]>
            missing_sequence_type (seq(float) input) => (seq(float) output) {
                output = SequenceMap (input) <
                    body = body (float body_input) => (float body_output) {
                        body_output = Identity(body_input)
                    }
                >
            }
            """
        )
        model.graph.input[0].ClearField("type")
        model.graph.output[0].ClearField("type")

        with pytest.raises(ValueError, match="Expected a sequence type"):
            inliner.inline_selected_functions(
                model, [], exclude=True, inline_schema_functions=True
            )

    def test_inline_ignores_constant_without_outputs(self):
        model = parser.parse_model(
            """
            <ir_version: 13, opset_import: [ "" : 13, "local" : 1 ]>
            constant_without_outputs (float[1] X) => (float[1] Y) {
                = Constant()
                Y = local.identity(X)
            }

            <opset_import: [ "" : 12 ], domain: "local">
            identity (X) => (Y) {
                Y = Identity(X)
            }
            """
        )

        inlined = inliner.inline_local_functions(model, convert_version=True)

        assert [node.op_type for node in inlined.graph.node] == [
            "Constant",
            "Identity",
        ]

    @pytest.mark.parametrize("selected", [False, True])
    @pytest.mark.parametrize("supply_default", [False, True])
    @pytest.mark.parametrize(
        "attribute_name,default,override,elem_type",
        [
            ("value_float", 7.0, 0.0, TensorProto.FLOAT),
            ("value_int", 7, 0, TensorProto.INT64),
            ("value_string", "default", "", TensorProto.STRING),
            (
                "value",
                helper.make_tensor("default", TensorProto.FLOAT, [1], [7.0]),
                helper.make_tensor("override", TensorProto.FLOAT, [1], [11.0]),
                TensorProto.FLOAT,
            ),
            (
                "value",
                helper.make_tensor("default", TensorProto.FLOAT, [1], [7.0]),
                helper.make_tensor("empty", TensorProto.FLOAT, [0], []),
                TensorProto.FLOAT,
            ),
        ],
    )
    def test_default_attributes(
        self, selected, supply_default, attribute_name, default, override, elem_type
    ):
        default_attr = helper.make_attribute("parameter", default)
        constant = helper.make_node("Constant", [], ["Y"])
        constant.attribute.append(
            helper.make_attribute_ref(
                attribute_name, default_attr.type, ref_attr_name="parameter"
            )
        )
        function = helper.make_function(
            "local",
            "constant",
            [],
            ["Y"],
            [constant],
            [helper.make_opsetid("", 18)],
            attribute_protos=[default_attr],
        )
        values = [default, default, override]
        calls = [
            helper.make_node("constant", [], [f"Y{i}"], domain="local")
            for i in range(len(values))
        ]
        # Explicitly supplying the default and falsey/empty values must override it.
        for i, (call, value) in enumerate(zip(calls, values, strict=True)):
            if i or supply_default:
                call.attribute.append(helper.make_attribute("parameter", value))
        expected = [
            numpy_helper.to_array(value)
            if isinstance(value, TensorProto)
            else np.asarray(value)
            for value in values
        ]
        model = helper.make_model(
            helper.make_graph(
                calls,
                "defaults",
                [],
                [
                    helper.make_tensor_value_info(f"Y{i}", elem_type, value.shape)
                    for i, value in enumerate(expected)
                ],
            ),
            functions=[function],
            ir_version=10,
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("local", 1),
            ],
        )
        checker.check_model(model, full_check=True)
        original = model.SerializeToString()
        result = (
            inliner.inline_selected_functions(model, [("local", "constant")])
            if selected
            else inliner.inline_local_functions(model)
        )
        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        assert len(result.graph.node) == len(calls)
        for node in result.graph.node:
            assert node.op_type == "Constant"
            assert len(node.attribute) == 1
            assert node.attribute[0].name == attribute_name
            assert not node.attribute[0].ref_attr_name
        for actual, value in zip(
            ReferenceEvaluator(result).run(None, {}), expected, strict=True
        ):
            np.testing.assert_array_equal(actual, value)

    @pytest.mark.parametrize("selected", [False, True])
    @pytest.mark.parametrize("missing_first", [False, True])
    def test_default_and_missing_attributes(self, selected, missing_first):
        activation = helper.make_node("HardSigmoid", ["X"], ["Y"])
        references = [
            helper.make_attribute_ref(
                "alpha", AttributeProto.FLOAT, ref_attr_name="slope"
            ),
            helper.make_attribute_ref(
                "beta", AttributeProto.FLOAT, ref_attr_name="offset"
            ),
        ]
        activation.attribute.extend(
            reversed(references) if missing_first else references
        )
        function = helper.make_function(
            "local",
            "activation",
            ["X"],
            ["Y"],
            [activation],
            [helper.make_opsetid("", 18)],
            attributes=["offset"],
            attribute_protos=[helper.make_attribute("slope", 0.25)],
        )
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("activation", ["X"], ["Y"], domain="local")],
                "mixed_attributes",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3])],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
            ),
            functions=[function],
            ir_version=10,
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("local", 1),
            ],
        )
        checker.check_model(model, full_check=True)
        original = model.SerializeToString()
        result = (
            inliner.inline_selected_functions(model, [("local", "activation")])
            if selected
            else inliner.inline_local_functions(model)
        )
        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        assert len(result.graph.node) == 1
        node = result.graph.node[0]
        assert node.op_type == "HardSigmoid"
        assert len(node.attribute) == 1
        assert node.attribute[0].name == "alpha"
        assert node.attribute[0].f == 0.25
        assert not node.attribute[0].ref_attr_name
        (actual,) = ReferenceEvaluator(result).run(
            None, {"X": np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)}
        )
        np.testing.assert_array_equal(actual, [0.25, 0.5, 0.75])

    @pytest.mark.parametrize("selected", [False, True])
    def test_default_graph_attribute(self, selected):
        def branch(value):
            return helper.make_graph(
                [
                    helper.make_node(
                        "Constant",
                        [],
                        ["temp"],
                        value=helper.make_tensor(
                            "value", TensorProto.FLOAT, [1], [value]
                        ),
                    ),
                    helper.make_node("Identity", ["temp"], ["out"]),
                ],
                "branch",
                [],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
            )

        conditional = helper.make_node("If", ["cond"], ["Y"])
        conditional.attribute.extend(
            helper.make_attribute_ref(name, AttributeProto.GRAPH, ref_attr_name="body")
            for name in ["then_branch", "else_branch"]
        )
        function = helper.make_function(
            "local",
            "conditional",
            ["cond"],
            ["Y"],
            [conditional],
            [helper.make_opsetid("", 18)],
            attribute_protos=[helper.make_attribute("body", branch(7.0))],
        )
        calls = [
            helper.make_node("conditional", ["cond"], [f"Y{i}"], domain="local")
            for i in range(3)
        ]
        calls[1].attribute.append(helper.make_attribute("body", branch(11.0)))
        model = helper.make_model(
            helper.make_graph(
                calls,
                "defaults",
                [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
                [
                    helper.make_tensor_value_info(f"Y{i}", TensorProto.FLOAT, [1])
                    for i in range(3)
                ],
            ),
            functions=[function],
            ir_version=10,
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("local", 1),
            ],
        )
        checker.check_model(model, full_check=True)
        original = model.SerializeToString()
        result = (
            inliner.inline_selected_functions(model, [("local", "conditional")])
            if selected
            else inliner.inline_local_functions(model)
        )
        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        # Repeated calls must receive independent renamed copies of the graph.
        graph_outputs = [
            attr.g.output[0].name
            for node in result.graph.node
            for attr in node.attribute
        ]
        assert len(set(graph_outputs)) == 6
        for condition in [True, False]:
            outputs = ReferenceEvaluator(result).run(
                None, {"cond": np.asarray(condition)}
            )
            for actual, value in zip(outputs, [7.0, 11.0, 7.0], strict=True):
                np.testing.assert_array_equal(
                    actual, np.asarray([value], dtype=np.float32)
                )

    @pytest.mark.parametrize("selected", [False, True])
    def test_graph_attribute_preserves_caller_scope(self, selected):
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 18, "local" : 1 ]>
            main (bool cond) => (float[1] Y) {
                Y = local.outer(cond)
            }

            <opset_import: [ "" : 18, "local" : 1 ], domain: "local">
            outer <slope: float = 0.25> (cond) => (Y) {
                Y = local.inner <
                    body = branch () => (float[1] out) {
                        X = Constant <value = float[1] {-2}> ()
                        out = LeakyRelu <alpha: float = @slope> (X)
                    }
                > (cond)
            }

            <opset_import: [ "" : 18 ], domain: "local">
            inner <
                slope: float = 0.75,
                body: graph = default_branch () => (float[1] out) {
                    out = Constant <value = float[1] {7}> ()
                }
            > (cond) => (Y) {
                Y = If <then_branch: graph = @body, else_branch: graph = @body> (cond)
            }
            """
        )
        checker.check_model(model, full_check=True)
        original = model.SerializeToString()
        if selected:
            result = inliner.inline_selected_functions(model, [("local", "inner")])
            checker.check_model(result, full_check=True)
            assert len(result.functions) == 1
            outer = result.functions[0]
            assert outer.name == "outer"
            assert outer.node[0].op_type == "If"
            # The copied graphs still belong to outer's attribute scope.
            for attr in outer.node[0].attribute:
                alpha = attr.g.node[1].attribute[0]
                assert alpha.name == "alpha"
                assert alpha.ref_attr_name == "slope"
                assert not alpha.HasField("f")
            result = inliner.inline_local_functions(result)
        else:
            result = inliner.inline_local_functions(model)
        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        for attr in result.graph.node[0].attribute:
            alpha = attr.g.node[1].attribute[0]
            assert not alpha.ref_attr_name
            assert alpha.f == 0.25
        for condition in [True, False]:
            (actual,) = ReferenceEvaluator(result).run(
                None, {"cond": np.asarray(condition)}
            )
            np.testing.assert_array_equal(actual, np.asarray([-0.5], dtype=np.float32))

    @pytest.mark.parametrize("selected", [False, True])
    @pytest.mark.parametrize("default", [None, 0.25])
    def test_nested_attribute_binding(self, selected, default):
        # The branch captures a function input and references an optional parameter.
        leaky_relu = helper.make_node("LeakyRelu", ["X"], ["out"])
        leaky_relu.attribute.append(
            helper.make_attribute_ref(
                "alpha", AttributeProto.FLOAT, ref_attr_name="slope"
            )
        )
        branch = helper.make_graph(
            [leaky_relu],
            "branch",
            [],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
        )
        conditional = helper.make_node(
            "If", ["cond"], ["Y"], then_branch=branch, else_branch=branch
        )
        function = helper.make_function(
            "local",
            "activation",
            ["X", "cond"],
            ["Y"],
            [conditional],
            [helper.make_opsetid("", 18)],
            attributes=["slope"] if default is None else [],
            attribute_protos=[]
            if default is None
            else [helper.make_attribute("slope", default)],
        )
        calls = [
            helper.make_node("activation", ["X", "cond"], [f"Y{i}"], domain="local")
            for i in range(2)
        ]
        calls[1].attribute.append(helper.make_attribute("slope", 0.0))
        model = helper.make_model(
            helper.make_graph(
                calls,
                "defaults",
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]),
                    helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                ],
                [
                    helper.make_tensor_value_info(f"Y{i}", TensorProto.FLOAT, [1])
                    for i in range(2)
                ],
            ),
            functions=[function],
            ir_version=10,
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("local", 1),
            ],
        )
        checker.check_model(model, full_check=True)
        original = model.SerializeToString()
        result = (
            inliner.inline_selected_functions(model, [("local", "activation")])
            if selected
            else inliner.inline_local_functions(model)
        )
        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        for node in result.graph.node:
            for attr in node.attribute:
                bound = attr.g.node[0]
                assert bound.input[0] == "X"
                assert all(not value.ref_attr_name for value in bound.attribute)
        if default is None:
            assert all(
                not attr.g.node[0].attribute for attr in result.graph.node[0].attribute
            )
        for condition in [True, False]:
            outputs = ReferenceEvaluator(result).run(
                None,
                {
                    "X": np.asarray([-2.0], dtype=np.float32),
                    "cond": np.asarray(condition),
                },
            )
            for actual, value in zip(
                outputs, [-2 * (0.01 if default is None else default), 0], strict=True
            ):
                np.testing.assert_allclose(actual, [value])
