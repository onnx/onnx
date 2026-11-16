# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import TensorProto, checker, defs, helper, inliner, parser
from onnx.reference import ReferenceEvaluator


class TestInliner:
    @pytest.mark.parametrize("selected", [False, True])
    def test_sharding_tensor_references(self, selected):
        model = parser.parse_model(
            """
            <ir_version: 11, opset_import: ["" : 17, "local" : 1]>
            g (float[2] A) => (float[2] out) {
                local__1 = local.F(A)
                out = local.F(local__1)
            }
            <domain: "local", opset_import: ["" : 17]>
            F (X) => (Y) {
                local = Identity(X)
                Y = Identity(local)
            }
            """
        )
        for name in ("two_devices", "replicated"):
            model.configuration.add(name=name, num_devices=2, device=["d0", "d1"])
        for node in model.functions[0].node:
            for name in ("two_devices", "replicated"):
                config = node.device_configurations.add(
                    configuration_id=name, pipeline_stage=1
                )
                for tensor_name in (*node.input, *node.output):
                    spec = config.sharding_spec.add(
                        tensor_name=tensor_name, device=[0, 1]
                    )
                    spec.index_to_device_group_map.add(key=0, value=[0, 1])
                    spec.sharded_dim.add(axis=0).simple_sharding.add(
                        dim_value=2, num_shards=2
                    )
        checker.check_model(model)

        if selected:
            inlined = inliner.inline_selected_functions(model, [("local", "F")])
        else:
            inlined = inliner.inline_local_functions(model)

        assert not inlined.functions
        assert len(inlined.graph.node) == 4
        assert inlined.graph.node[0].input[0] == "A"
        assert inlined.graph.node[-1].output[0] == "out"
        assert inlined.graph.node[0].output[0] != "local__1"
        for index, node in enumerate(inlined.graph.node):
            original = model.functions[0].node[index % 2]
            for config, original_config in zip(
                node.device_configurations, original.device_configurations, strict=True
            ):
                expected = type(config)()
                expected.CopyFrom(original_config)
                for spec, tensor_name in zip(
                    expected.sharding_spec, (*node.input, *node.output), strict=True
                ):
                    spec.tensor_name = tensor_name
                assert config == expected
        checker.check_model(inlined)

    def test_sharding_tensor_references_in_subgraphs(self):
        model = parser.parse_model(
            """
            <ir_version: 11, opset_import: ["" : 17, "local" : 1]>
            g (bool cond, float[2] A) => (float[2] out) {
                out = local.F(cond, A)
            }
            <domain: "local", opset_import: ["" : 17]>
            F (C, X) => (Y) {
                Y = If(C) <
                    then_branch = then_graph () => (float[2] result) {
                        local = Identity(X)
                        result = If(C) <
                            then_branch = inner_then () => (float[2] value) {
                                value = Identity(local)
                            },
                            else_branch = inner_else () => (float[2] value) {
                                value = Identity(local)
                            }
                        >
                    },
                    else_branch = else_graph () => (float[2] local) {
                        local = Identity(X)
                    }
                >
            }
            """
        )
        model.configuration.add(name="two_devices", num_devices=2)
        if_node = model.functions[0].node[0]
        branches = {attr.name: attr.g for attr in if_node.attribute}
        branches["else_branch"].initializer.append(
            helper.make_tensor("X", TensorProto.FLOAT, [2], [1.0, 2.0])
        )

        def add_sharding(node):
            config = node.device_configurations.add(configuration_id="two_devices")
            for tensor_name in (*node.input, *node.output):
                config.sharding_spec.add(tensor_name=tensor_name, device=[0, 1])
            for attr in node.attribute:
                if attr.HasField("g"):
                    for child in attr.g.node:
                        add_sharding(child)

        add_sharding(if_node)
        checker.check_model(model)
        inlined = inliner.inline_local_functions(model)
        renamed_if = inlined.graph.node[0]
        renamed_branches = {attr.name: attr.g for attr in renamed_if.attribute}
        assert renamed_branches["then_branch"].node[0].input[0] == "A"
        shadowed = renamed_branches["else_branch"]
        assert shadowed.node[0].input[0] == shadowed.initializer[0].name
        assert shadowed.node[0].input[0] != "A"
        outer_local = renamed_branches["then_branch"].node[0].output[0]
        for attr in renamed_branches["then_branch"].node[1].attribute:
            assert attr.g.node[0].input[0] == outer_local

        def check_sharding(node):
            specs = node.device_configurations[0].sharding_spec
            assert [spec.tensor_name for spec in specs] == [*node.input, *node.output]
            assert all(list(spec.device) == [0, 1] for spec in specs)
            for attr in node.attribute:
                if attr.HasField("g"):
                    for child in attr.g.node:
                        check_sharding(child)

        check_sharding(renamed_if)
        checker.check_model(inlined)

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
