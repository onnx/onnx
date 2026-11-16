# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import checker, defs, inliner, parser
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
