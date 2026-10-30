# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import onnx.version_converter
from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator


class TestVersionConverter:
    @pytest.mark.parametrize("target_version", [12, 14])
    def test_reject_incompatible_function_schemas(self, target_version: int) -> None:
        function = helper.make_function(
            "local",
            "IdentityFn",
            ["X"],
            ["Y"],
            [helper.make_node("Identity", ["X"], ["Y"])],
            [helper.make_opsetid("", 13)],
        )
        graph = helper.make_graph(
            [helper.make_node("IdentityFn", ["X"], ["Y"], domain="local")],
            "function_schema_boundary",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(
            graph,
            functions=[function],
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("local", 1),
            ],
        )
        checker.check_model(model)
        original = model.SerializeToString()
        with pytest.raises(
            onnx.version_converter.ConvertError,
            match="Cannot preserve model-local functions at the requested opset",
        ):
            onnx.version_converter.convert_version(model, target_version)
        assert model.SerializeToString() == original

    def test_preserve_function_with_unloaded_external_constant(self) -> None:
        tensor = TensorProto(name="c", data_type=TensorProto.FLOAT, dims=[1])
        tensor.data_location = TensorProto.EXTERNAL
        for key, value in (
            ("location", "not_in_working_directory.bin"),
            ("offset", "0"),
            ("length", "4"),
        ):
            entry = tensor.external_data.add()
            entry.key = key
            entry.value = value
        function = helper.make_function(
            "local",
            "ConstFn",
            ["X"],
            ["Y"],
            [
                helper.make_node("Constant", [], ["C"], value=tensor),
                helper.make_node("Abs", ["C"], ["Y"]),
            ],
            [helper.make_opsetid("", 13)],
        )
        graph = helper.make_graph(
            [helper.make_node("ConstFn", ["X"], ["Y"], domain="local")],
            "external_constant_function",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(
            graph,
            functions=[function],
            opset_imports=[
                helper.make_opsetid("", 13),
                helper.make_opsetid("local", 1),
            ],
        )
        converted = onnx.version_converter.convert_version(model, 14)
        assert list(converted.functions) == list(model.functions)

    @pytest.mark.parametrize("target_version", [13, 14, 15])
    def test_preserve_local_functions(self, target_version: int) -> None:
        function = helper.make_function(
            "local",
            "AbsFn",
            ["X"],
            ["Y"],
            [helper.make_node("Abs", ["X"], ["Y"])],
            [helper.make_opsetid("", 13)],
        )
        graph = helper.make_graph(
            [helper.make_node("AbsFn", ["X"], ["Y"], domain="local")],
            "local_function",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(
            graph,
            functions=[function],
            opset_imports=[
                helper.make_opsetid("", 14),
                helper.make_opsetid("local", 1),
            ],
        )
        checker.check_model(model)
        converted = onnx.version_converter.convert_version(model, target_version)
        assert list(converted.functions) == list(model.functions)
        assert converted.opset_import[0].version == target_version
        checker.check_model(converted)
        feeds = {"X": np.array([2.0], dtype=np.float32)}
        np.testing.assert_array_equal(
            ReferenceEvaluator(converted).run(None, feeds)[0], feeds["X"]
        )
