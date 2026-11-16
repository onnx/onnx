# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import (
    AttributeProto,
    TensorProto,
    checker,
    defs,
    helper,
    inliner,
    numpy_helper,
    parser,
    shape_inference,
)
from onnx.reference import ReferenceEvaluator


class TestInliner:
    def test_groupedmatmul_linked_transb(self):
        model = parser.parse_model(
            """
            <ir_version: 13, opset_import: ["" : 29, "local" : 1]>
            g (float[1,2] X, float[1,2,2] W, int64[1,1] I) => (float[1,1,2] Y) {
                Y = local.project <transpose = 1> (X, W, I)
            }
            <domain: "local", opset_import: ["" : 29]>
            project <transpose> (float[1,2] X, float[1,2,2] W, int64[1,1] I)
                => (float[1,1,2] Y) {
                Y = GroupedMatMul <transB: int = @transpose> (X, W, I)
            }
            """
        )
        node = model.functions[0].node[0]
        schema = defs.get_schema("GroupedMatMul", 29)
        input_types = [value.type.SerializeToString() for value in model.graph.input]
        assert not schema.get_context_dependent_function(
            node.SerializeToString(), input_types
        )
        inlined = inliner.inline_selected_functions(
            model,
            [("local", "project")],
            exclude=True,
            inline_schema_functions=True,
        )
        retained_node = inlined.functions[0].node[0]
        assert retained_node.op_type == "GroupedMatMul"
        assert retained_node.attribute[0].ref_attr_name == "transpose"
        assert not retained_node.attribute[0].HasField("i")
        checker.check_model(inlined)
        feeds = {
            "X": np.array([[1, 2]], dtype=np.float32),
            "W": np.array([[[1, 3], [5, 7]]], dtype=np.float32),
            "I": np.array([[0]], dtype=np.int64),
        }
        expected = np.array([[[7, 19]]], dtype=np.float32)
        for candidate in (model, inlined):
            evaluator = ReferenceEvaluator(candidate)
            np.testing.assert_array_equal(evaluator.run(None, feeds)[0], expected)
            function = evaluator.functions_[("local", "project")]
            for transpose in (0, 1, 2, 0):
                result = function.run(None, feeds, attributes={"transpose": transpose})
                expected_call = (
                    expected if transpose else np.array([[[11, 17]]], dtype=np.float32)
                )
                np.testing.assert_array_equal(result[0], expected_call)
        bound = inliner.inline_selected_functions(
            inlined, [], exclude=True, inline_schema_functions=True
        )
        assert not bound.functions
        assert all(node.op_type != "GroupedMatMul" for node in bound.graph.node)
        assert any(node.op_type == "Transpose" for node in bound.graph.node)
        checker.check_model(bound)
        np.testing.assert_array_equal(
            ReferenceEvaluator(bound).run(None, feeds)[0], expected
        )

    def test_groupedmatmul_implicit_zero_transb(self):
        model = parser.parse_model(
            """
            <ir_version: 13, opset_import: ["" : 29]>
            g (float[1,2] X, float[1,2,2] W, int64[1,1] I) => (float[1,1,2] Y) {
                Y = GroupedMatMul (X, W, I)
            }
            """
        )
        model.graph.node[0].attribute.add(name="transB", type=2)
        checker.check_model(model)
        feeds = {
            "X": np.array([[1, 2]], dtype=np.float32),
            "W": np.array([[[1, 3], [5, 7]]], dtype=np.float32),
            "I": np.array([[0]], dtype=np.int64),
        }
        for candidate in (
            model,
            inliner.inline_selected_functions(
                model, [], exclude=True, inline_schema_functions=True
            ),
        ):
            np.testing.assert_array_equal(
                ReferenceEvaluator(candidate).run(None, feeds)[0],
                np.array([[[11, 17]]], dtype=np.float32),
            )

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

    @pytest.mark.parametrize("selected", [False, True])
    def test_default_graph_nested_attribute_binding(self, selected):
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 18, "local" : 1 ]>
            main (float[3] X, bool cond) => (float[3] Y0, float[3] Y1, float[3] Y2) {
                Y0 = local.conditional(X, cond)
                Y1 = local.conditional <slope = 0.25, offset = 0.0> (X, cond)
                Y2 = local.conditional <slope = 0.0> (X, cond)
            }

            <opset_import: [ "" : 18 ], domain: "local">
            conditional <
                offset,
                slope: float = 0.75,
                body: graph = default_branch () => (float[3] out) {
                    out = HardSigmoid <alpha: float = @slope, beta: float = @offset> (X)
                }
            > (X, cond) => (Y) {
                Y = If <then_branch: graph = @body, else_branch: graph = @body> (cond)
            }
            """
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
        assert len(result.graph.node) == 3
        for node in result.graph.node:
            for attr in node.attribute:
                bound = attr.g.node[0]
                assert bound.input[0] == "X"
                assert all(not value.ref_attr_name for value in bound.attribute)
        assert all(
            len(attr.g.node[0].attribute) == 1
            for attr in result.graph.node[0].attribute
        )
        assert all(
            len(attr.g.node[0].attribute) == 2
            for attr in result.graph.node[1].attribute
        )
        x = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
        expected = [
            np.clip(0.75 * x + 0.5, 0.0, 1.0),
            np.clip(0.25 * x + 0.0, 0.0, 1.0),
            np.clip(0.0 * x + 0.5, 0.0, 1.0),
        ]
        for condition in [True, False]:
            outputs = ReferenceEvaluator(result).run(
                None, {"X": x, "cond": np.asarray(condition)}
            )
            for actual, value in zip(outputs, expected, strict=True):
                np.testing.assert_allclose(actual, value)

    @pytest.mark.parametrize("selected", [False, True])
    def test_default_graph_preserves_callee_scope(self, selected):
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 18, "local" : 1 ]>
            main (bool cond) => (float[1] Y) {
                Y = local.outer(cond)
            }

            <opset_import: [ "" : 18, "local" : 1 ], domain: "local">
            outer <slope: float = 0.25> (cond) => (Y) {
                Y = local.inner(cond)
            }

            <opset_import: [ "" : 18 ], domain: "local">
            inner <
                slope: float = 0.75,
                body: graph = default_branch () => (float[1] out) {
                    X = Constant <value = float[1] {-2}> ()
                    out = LeakyRelu <alpha: float = @slope> (X)
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
            # Default graphs must bind in inner's scope rather than escaping to outer.
            for attr in outer.node[0].attribute:
                alpha = attr.g.node[1].attribute[0]
                assert alpha.name == "alpha"
                assert not alpha.ref_attr_name
                assert alpha.f == 0.75
            result = inliner.inline_local_functions(result)
        else:
            result = inliner.inline_local_functions(model)
        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        for attr in result.graph.node[0].attribute:
            alpha = attr.g.node[1].attribute[0]
            assert not alpha.ref_attr_name
            assert alpha.f == 0.75
        for condition in [True, False]:
            (actual,) = ReferenceEvaluator(result).run(
                None, {"cond": np.asarray(condition)}
            )
            np.testing.assert_array_equal(actual, np.asarray([-1.5], dtype=np.float32))

    @pytest.mark.parametrize("selected", [False, True])
    def test_default_graph_dag_reuse(self, selected):
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 18, "local" : 1 ]>
            main (bool cond) => (float[1] Y) {
                Y = local.nested(cond)
            }

            <opset_import: [ "" : 18 ], domain: "local">
            nested <
                val: tensor = float[1] {7},
                leaf: graph = leaf_branch () => (float[1] out) {
                    out = Constant <value: tensor = @val> ()
                },
                body: graph = outer_branch () => (float[1] out) {
                    out = If <then_branch: graph = @leaf, else_branch: graph = @leaf> (cond)
                }
            > (cond) => (Y) {
                Y = If <then_branch: graph = @body, else_branch: graph = @body> (cond)
            }
            """
        )
        checker.check_model(model, full_check=True)
        result = (
            inliner.inline_selected_functions(model, [("local", "nested")])
            if selected
            else inliner.inline_local_functions(model)
        )
        checker.check_model(result, full_check=True)
        for condition in [True, False]:
            (actual,) = ReferenceEvaluator(result).run(
                None, {"cond": np.asarray(condition)}
            )
            np.testing.assert_array_equal(actual, np.asarray([7.0], dtype=np.float32))

    @pytest.mark.parametrize("selected", [False, True])
    @pytest.mark.parametrize("mutual", [False, True])
    def test_default_graph_rejects_cycle(self, selected, mutual):
        leaf = helper.make_graph(
            [
                helper.make_node(
                    "Constant",
                    [],
                    ["out"],
                    value=helper.make_tensor("v", TensorProto.FLOAT, [1], [3.0]),
                )
            ],
            "leaf",
            [],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
        )

        def cyclic_branch(name, target_ref):
            inner_if = helper.make_node("If", ["cond"], ["out"])
            inner_if.attribute.extend(
                [
                    helper.make_attribute_ref(
                        "then_branch", AttributeProto.GRAPH, ref_attr_name=target_ref
                    ),
                    helper.make_attribute("else_branch", leaf),
                ]
            )
            return helper.make_graph(
                [inner_if],
                name,
                [],
                [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
            )

        defaults = (
            [
                helper.make_attribute("body", cyclic_branch("branch_a", "other")),
                helper.make_attribute("other", cyclic_branch("branch_b", "body")),
            ]
            if mutual
            else [helper.make_attribute("body", cyclic_branch("self_branch", "body"))]
        )
        top_if = helper.make_node("If", ["cond"], ["Y"])
        top_if.attribute.extend(
            helper.make_attribute_ref(name, AttributeProto.GRAPH, ref_attr_name="body")
            for name in ["then_branch", "else_branch"]
        )
        function = helper.make_function(
            "local",
            "cyclic_defaults",
            ["cond"],
            ["Y"],
            [top_if],
            [helper.make_opsetid("", 18)],
            attribute_protos=defaults,
        )

        def make_model(override_body=False):
            call = helper.make_node("cyclic_defaults", ["cond"], ["Y"], domain="local")
            if override_body:
                call.attribute.append(helper.make_attribute("body", leaf))
            return helper.make_model(
                helper.make_graph(
                    [call],
                    "main",
                    [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
                    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
                ),
                functions=[function],
                ir_version=10,
                opset_imports=[
                    helper.make_opsetid("", 18),
                    helper.make_opsetid("local", 1),
                ],
            )

        def inline(target):
            return (
                inliner.inline_selected_functions(
                    target, [("local", "cyclic_defaults")]
                )
                if selected
                else inliner.inline_local_functions(target)
            )

        def check_full(target):
            checker.check_model(target, full_check=True)

        model = make_model()
        with pytest.raises(shape_inference.InferenceError, match="Cycle detected"):
            check_full(model)
        with pytest.raises(checker.ValidationError, match="Cycle detected"):
            inline(model)

        # Overriding the cyclic parameter at the call site avoids expanding the cyclic default.
        overridden = make_model(override_body=True)
        checker.check_model(overridden, full_check=True)
        result = inline(overridden)
        checker.check_model(result, full_check=True)
        (actual,) = ReferenceEvaluator(result).run(None, {"cond": np.asarray(True)})
        np.testing.assert_array_equal(actual, np.asarray([3.0], dtype=np.float32))

    @pytest.mark.parametrize("selected", [False, True])
    def test_graph_value_scope_isolation(self, selected):
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 18, "local" : 1 ]>
            main (bool cond) => (float[12] Y_nested, float[3] Y_direct, float[5] Y_caller_local) {
                X = Constant <value = float[3] {1, 2, 3}> ()
                A = Constant <value = float[7] {10, 11, 12, 13, 14, 15, 16}> ()
                L = Constant <value = float[5] {20, 21, 22, 23, 24}> ()
                Y_nested, Y_direct = local.callee <
                    sub = caller_branch () => (float[?] out) {
                        local_copy = Identity(X)
                        out = Identity(local_copy)
                    }
                > (A, cond)
                Y_caller_local = Identity(L)
            }

            <opset_import: [ "" : 18 ], domain: "local">
            callee <
                sub,
                body: graph = default_branch () => (float[?] out) {
                    L = Constant <value = float[2] {90, 91}> ()
                    from_caller = If <then_branch: graph = @sub, else_branch: graph = @sub> (cond)
                    from_callee = Identity(X)
                    out = Concat <axis = 0> (from_caller, from_callee, L)
                }
            > (X, cond) => (Y_nested, Y_direct) {
                Y_nested = If <then_branch: graph = @body, else_branch: graph = @body> (cond)
                Y_direct = If <then_branch: graph = @sub, else_branch: graph = @sub> (cond)
            }
            """
        )
        inferred = shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True
        )
        assert [
            d.dim_value for d in inferred.graph.output[0].type.tensor_type.shape.dim
        ] == [12]
        assert [
            d.dim_value for d in inferred.graph.output[1].type.tensor_type.shape.dim
        ] == [3]
        checker.check_model(model, full_check=True)
        original = model.SerializeToString()
        result = (
            inliner.inline_selected_functions(model, [("local", "callee")])
            if selected
            else inliner.inline_local_functions(model)
        )
        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        for condition in [True, False]:
            nested_out, direct_out, caller_local_out = ReferenceEvaluator(result).run(
                None, {"cond": np.asarray(condition)}
            )
            np.testing.assert_array_equal(
                nested_out,
                np.asarray(
                    [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 90, 91], dtype=np.float32
                ),
            )
            np.testing.assert_array_equal(
                direct_out, np.asarray([1, 2, 3], dtype=np.float32)
            )
            np.testing.assert_array_equal(
                caller_local_out, np.asarray([20, 21, 22, 23, 24], dtype=np.float32)
            )

    @pytest.mark.parametrize("selected", [False, True])
    def test_nested_function_graph_value_and_attribute_scope(self, selected):
        model = parser.parse_model(
            """
            <ir_version: 10, opset_import: [ "" : 18, "local" : 1 ]>
            main (bool cond) => (float[15] Y) {
                X = Constant <value = float[3] {-4, -8, -12}> ()
                M = Constant <value = float[5] {-4, -4, -4, -4, -4}> ()
                N = Constant <value = float[7] {-8, -8, -8, -8, -8, -8, -8}> ()
                Y = local.outer <
                    g_from_main = main_branch () => (float[?] m_out) {
                        m_out = Identity(X)
                    }
                > (M, N, cond)
            }

            <opset_import: [ "" : 18, "local" : 1 ], domain: "local">
            outer <
                g_from_main,
                slope: float = 0.25,
                outer_body: graph = outer_default () => (float[?] out) {
                    from_main = If <then_branch: graph = @g_from_main, else_branch: graph = @g_from_main> (cond)
                    from_outer = LeakyRelu <alpha: float = @slope> (X)
                    out = Concat <axis = 0> (from_main, from_outer)
                }
            > (X, B, cond) => (Y) {
                Y = local.inner <sub: graph = @outer_body> (B, cond)
            }

            <opset_import: [ "" : 18 ], domain: "local">
            inner <
                sub,
                slope: float = 0.75,
                body: graph = default_branch () => (float[?] out) {
                    from_caller = If <then_branch: graph = @sub, else_branch: graph = @sub> (cond)
                    from_callee = LeakyRelu <alpha: float = @slope> (X)
                    out = Concat <axis = 0> (from_caller, from_callee)
                }
            > (X, cond) => (Y) {
                Y = If <then_branch: graph = @body, else_branch: graph = @body> (cond)
            }
            """
        )
        inferred = shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True
        )
        assert [
            d.dim_value for d in inferred.graph.output[0].type.tensor_type.shape.dim
        ] == [15]
        checker.check_model(model, full_check=True)
        original = model.SerializeToString()
        if selected:
            partial_inner = inliner.inline_selected_functions(
                model, [("local", "inner")]
            )
            checker.check_model(partial_inner, full_check=True)
            partial_outer = inliner.inline_selected_functions(
                model, [("local", "outer")]
            )
            checker.check_model(partial_outer, full_check=True)
            result = inliner.inline_local_functions(partial_inner)
        else:
            result = inliner.inline_local_functions(model)
        assert model.SerializeToString() == original
        checker.check_model(result, full_check=True)
        expected = np.asarray(
            [-4, -8, -12, -1, -1, -1, -1, -1, -6, -6, -6, -6, -6, -6, -6],
            dtype=np.float32,
        )
        for condition in [True, False]:
            (actual,) = ReferenceEvaluator(result).run(
                None, {"cond": np.asarray(condition)}
            )
            np.testing.assert_allclose(actual, expected)
