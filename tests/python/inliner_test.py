# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import checker, helper, inliner, numpy_helper, parser
from onnx.reference import ReferenceEvaluator


class TestInliner:
    def test_nested_sparse_initializer_collision(self):
        model = parser.parse_model(
            """
            <ir_version: 13, opset_import: ["" : 29, "local" : 1]>
            graph (float[2,1] X) => (float[2,1] Y) { Y = local.scan(X) }
            <opset_import: ["" : 29], domain: "local">
            scan (x) => (y) {
                y = Scan (x) <
                    num_scan_inputs = 1,
                    body = step (float[1] item) => (float[1] result) {
                        temp = Identity(item)
                        result = Add(temp, temp__1)
                    }
                >
            }
            """
        )
        body = next(
            attr.g
            for attr in model.functions[0].node[0].attribute
            if attr.name == "body"
        )
        body.sparse_initializer.append(
            helper.make_sparse_tensor(
                numpy_helper.from_array(np.array([7.0], dtype=np.float32), "temp__1"),
                numpy_helper.from_array(np.array([0], dtype=np.int64)),
                [1],
            )
        )
        checker.check_model(model)
        expanded = inliner.inline_local_functions(model)
        checker.check_model(expanded)
        expanded_body = next(
            attr.g for attr in expanded.graph.node[0].attribute if attr.name == "body"
        )
        initializer = expanded_body.sparse_initializer[0]
        assert initializer.values.name != expanded_body.node[0].output[0]
        assert initializer.values.name == expanded_body.node[1].input[1]
        assert initializer.values.raw_data == body.sparse_initializer[0].values.raw_data
        assert initializer.indices == body.sparse_initializer[0].indices

    def test_scan_initializer_preserves_input_override(self):
        model = parser.parse_model(
            """
            <ir_version: 13, opset_import: ["" : 29, "local" : 1]>
            graph (float[2,1] X) => (float[2,1] Y) { Y = local.scan(X) }
            <opset_import: ["" : 29], domain: "local">
            scan (x) => (y) {
                y = Scan (x) <
                    num_scan_inputs = 1,
                    body = step (float[1] item) => (float[1] result) {
                        result = Identity(item)
                    }
                >
            }
            """
        )
        body = next(
            attr.g
            for attr in model.functions[0].node[0].attribute
            if attr.name == "body"
        )
        body.initializer.append(
            numpy_helper.from_array(np.array([99.0], dtype=np.float32), name="item")
        )
        checker.check_model(model)
        feeds = {"X": np.array([[2.0], [3.0]], dtype=np.float32)}
        np.testing.assert_array_equal(
            ReferenceEvaluator(model).run(None, feeds)[0], feeds["X"]
        )
        expanded = inliner.inline_local_functions(model)
        checker.check_model(expanded)
        np.testing.assert_array_equal(
            ReferenceEvaluator(expanded).run(None, feeds)[0], feeds["X"]
        )

    @pytest.mark.parametrize("extra_output", ["", "Unexpected"])
    def test_trailing_optional_output_binding(self, extra_output):
        model = parser.parse_model(
            """
            <ir_version: 13, opset_import: ["" : 29, "local" : 1]>
            graph (float[1] X) => (float[1] Y) { Y = local.identity(X) }
            <opset_import: ["" : 29], domain: "local">
            identity (x) => (y) { y = Identity(x) }
            """
        )
        model.graph.node[0].output.append(extra_output)
        if extra_output:
            with pytest.raises(RuntimeError, match="actual parameters"):
                inliner.inline_local_functions(model)
        else:
            checker.check_model(inliner.inline_local_functions(model), full_check=True)

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
