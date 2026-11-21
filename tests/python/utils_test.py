# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile

import numpy as np
import pytest

import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


class TestUtilityFunctions:
    def test_extract_model(self) -> None:
        def create_tensor(name):
            return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])

        A0 = create_tensor("A0")
        A1 = create_tensor("A1")
        B0 = create_tensor("B0")
        B1 = create_tensor("B1")
        B2 = create_tensor("B2")
        C0 = create_tensor("C0")
        C1 = create_tensor("C1")
        D0 = create_tensor("D0")
        L0_0 = helper.make_node("Add", ["A0", "A1"], ["B0"])
        L0_1 = helper.make_node("Sub", ["A0", "A1"], ["B1"])
        L0_2 = helper.make_node("Mul", ["A0", "A1"], ["B2"])
        L1_0 = helper.make_node("Add", ["B0", "B1"], ["C0"])
        L1_1 = helper.make_node("Sub", ["B1", "B2"], ["C1"])
        L2_0 = helper.make_node("Mul", ["C0", "C1"], ["D0"])

        g0 = helper.make_graph(
            [L0_0, L0_1, L0_2, L1_0, L1_1, L2_0], "test", [A0, A1], [D0]
        )
        m0 = helper.make_model(g0, producer_name="test")
        tdir = tempfile.mkdtemp()
        p0 = os.path.join(tdir, "original.onnx")
        onnx.save(m0, p0)

        p1 = os.path.join(tdir, "extracted.onnx")
        input_names = ["B0", "B1", "B2"]
        output_names = ["C0", "C1"]
        onnx.utils.extract_model(p0, p1, input_names, output_names)

        m1 = onnx.load(p1)
        assert m1.producer_name == "onnx.utils.extract_model"
        assert m1.ir_version == m0.ir_version
        assert m1.opset_import == m0.opset_import
        assert len(m1.graph.node) == 2
        assert len(m1.graph.input) == 3
        assert len(m1.graph.output) == 2
        assert m1.graph.input[0] == B0
        assert m1.graph.input[1] == B1
        assert m1.graph.input[2] == B2
        assert m1.graph.output[0] == C0
        assert m1.graph.output[1] == C1
        shutil.rmtree(tdir, ignore_errors=True)

    def test_extract_model_keeps_values_used_only_in_subgraphs(self) -> None:
        # The If branches read "t" (produced by a node in the main graph) and
        # "w" (a main-graph initializer) from the outer scope. Neither is a
        # direct input of the If node, but both are needed by the extracted model.
        float_vec = [2]
        then_branch = helper.make_graph(
            [helper.make_node("Add", ["t", "w"], ["then_out"])],
            "then_branch",
            [],
            [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, float_vec)],
        )
        else_branch = helper.make_graph(
            [helper.make_node("Identity", ["t"], ["else_out"])],
            "else_branch",
            [],
            [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, float_vec)],
        )
        graph = helper.make_graph(
            [
                helper.make_node("Neg", ["X"], ["t"]),
                helper.make_node(
                    "If",
                    ["cond"],
                    ["Y"],
                    then_branch=then_branch,
                    else_branch=else_branch,
                ),
                helper.make_node("Relu", ["Y"], ["Z"]),
            ],
            "test",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, float_vec),
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            ],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, float_vec)],
            initializer=[
                helper.make_tensor("w", TensorProto.FLOAT, float_vec, [10.0, 20.0])
            ],
        )
        model = helper.make_model(graph, producer_name="test")
        onnx.checker.check_model(model, full_check=True)

        extracted = onnx.utils.Extractor(model).extract_model(["X", "cond"], ["Z"])

        assert [node.op_type for node in extracted.graph.node] == ["Neg", "If", "Relu"]
        assert [init.name for init in extracted.graph.initializer] == ["w"]
        onnx.checker.check_model(extracted, full_check=True)
        feeds = {"X": np.array([-1.0, -2.0], dtype=np.float32), "cond": np.array(True)}
        expected = ReferenceEvaluator(model).run(None, feeds)
        actual = ReferenceEvaluator(extracted).run(None, feeds)
        np.testing.assert_allclose(actual[0], expected[0])

    def test_extract_model_keeps_values_used_only_in_nested_subgraphs(self) -> None:
        # The initializer "w" is only read by an If nested inside another If.
        float_vec = [2]

        def branch(name, nodes, output):
            return helper.make_graph(
                nodes,
                name,
                [],
                [helper.make_tensor_value_info(output, TensorProto.FLOAT, float_vec)],
            )

        inner_if = helper.make_node(
            "If",
            ["cond"],
            ["inner_out"],
            then_branch=branch(
                "inner_then", [helper.make_node("Add", ["X", "w"], ["a"])], "a"
            ),
            else_branch=branch(
                "inner_else", [helper.make_node("Identity", ["X"], ["b"])], "b"
            ),
        )
        outer_if = helper.make_node(
            "If",
            ["cond"],
            ["Y"],
            then_branch=branch("outer_then", [inner_if], "inner_out"),
            else_branch=branch(
                "outer_else", [helper.make_node("Neg", ["X"], ["c"])], "c"
            ),
        )
        graph = helper.make_graph(
            [outer_if],
            "test",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, float_vec),
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            ],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, float_vec)],
            initializer=[
                helper.make_tensor("w", TensorProto.FLOAT, float_vec, [10.0, 20.0])
            ],
        )
        model = helper.make_model(graph, producer_name="test")
        onnx.checker.check_model(model, full_check=True)

        extracted = onnx.utils.Extractor(model).extract_model(["X", "cond"], ["Y"])

        assert [init.name for init in extracted.graph.initializer] == ["w"]
        onnx.checker.check_model(extracted, full_check=True)

    def test_extract_model_ignores_names_defined_inside_subgraphs(self) -> None:
        # "x" is local to the then branch. A main-graph node after the If also
        # outputs "x", which is allowed because it is not visible inside the
        # branch, so it must not be pulled into the extracted model.
        float_vec = [2]
        then_branch = helper.make_graph(
            [
                helper.make_node("Neg", ["A"], ["x"]),
                helper.make_node("Relu", ["x"], ["t"]),
            ],
            "then_branch",
            [],
            [helper.make_tensor_value_info("t", TensorProto.FLOAT, float_vec)],
        )
        else_branch = helper.make_graph(
            [helper.make_node("Identity", ["A"], ["e"])],
            "else_branch",
            [],
            [helper.make_tensor_value_info("e", TensorProto.FLOAT, float_vec)],
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["Y"],
                    then_branch=then_branch,
                    else_branch=else_branch,
                ),
                helper.make_node("Abs", ["B"], ["x"]),
            ],
            "test",
            [
                helper.make_tensor_value_info("A", TensorProto.FLOAT, float_vec),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, float_vec),
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            ],
            [
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, float_vec),
                helper.make_tensor_value_info("x", TensorProto.FLOAT, float_vec),
            ],
        )
        model = helper.make_model(graph, producer_name="test")
        onnx.checker.check_model(model, full_check=True)

        extracted = onnx.utils.Extractor(model).extract_model(["A", "cond"], ["Y"])

        assert [node.op_type for node in extracted.graph.node] == ["If"]
        onnx.checker.check_model(extracted, full_check=True)

    def test_tar_members_filter_rejects_sibling_prefix_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            base = os.path.join(tdir, "model")
            os.mkdir(base)
            tar_path = os.path.join(tdir, "payload.tar")

            with tarfile.open(tar_path, "w") as tar:
                payload = b"outside extraction root"
                info = tarfile.TarInfo("../model_evil/pwned.txt")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            with tarfile.open(tar_path) as tar:  # noqa: SIM117
                with pytest.raises(RuntimeError, match="directory traversal"):
                    onnx.utils._tar_members_filter(tar, base)
