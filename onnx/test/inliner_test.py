# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from onnx import checker, inliner, parser


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
