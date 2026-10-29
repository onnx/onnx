# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import onnx.version_converter
from onnx import TensorProto, checker, helper


class TestVersionConverter:
    @pytest.mark.parametrize("target_version", [13, 14, 15])
    def test_preserve_graph_and_node_metadata(self, target_version: int) -> None:
        branch = helper.make_graph(
            [helper.make_node("Abs", ["X"], ["B"])],
            "branch",
            [],
            [helper.make_tensor_value_info("B", TensorProto.FLOAT, [1])],
        )
        helper.set_metadata_props(branch, {"branch": "graph"})
        helper.set_metadata_props(branch.node[0], {"branch": "node"})
        graph = helper.make_graph(
            [
                helper.make_node(
                    "If", ["cond"], ["Y"], then_branch=branch, else_branch=branch
                )
            ],
            "metadata",
            [
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        )
        helper.set_metadata_props(graph, {"graph": "value"})
        helper.set_metadata_props(graph.node[0], {"node": "value"})
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        converted = onnx.version_converter.convert_version(model, target_version)
        checker.check_model(converted)
        assert converted.graph.metadata_props == model.graph.metadata_props
        assert converted.graph.node[0].metadata_props == graph.node[0].metadata_props
        for attr in converted.graph.node[0].attribute:
            assert attr.g.metadata_props == branch.metadata_props
            assert attr.g.node[0].metadata_props == branch.node[0].metadata_props

    def test_preserve_replaced_node_metadata(self) -> None:
        graph = helper.make_graph(
            [helper.make_node("Upsample", ["X", "scales"], ["Y"])],
            "replacement",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            [helper.make_tensor("scales", TensorProto.FLOAT, [1], [2.0])],
        )
        helper.set_metadata_props(graph.node[0], {"origin": "upsample"})
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 9)])
        converted = onnx.version_converter.convert_version(model, 10)
        checker.check_model(converted)
        assert converted.graph.node[0].op_type == "Resize"
        assert converted.graph.node[0].metadata_props == graph.node[0].metadata_props
