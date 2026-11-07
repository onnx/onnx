# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import onnx.version_converter
from onnx import TensorProto, checker, helper


class TestVersionConverter:
    @pytest.mark.parametrize("used", [False, True])
    @pytest.mark.parametrize("nested", [False, True])
    def test_reject_sparse_initializers(self, used: bool, nested: bool) -> None:
        sparse = helper.make_sparse_tensor(
            helper.make_tensor("S", TensorProto.FLOAT, [1], [2.0]),
            helper.make_tensor("indices", TensorProto.INT64, [1], [0]),
            [1],
        )
        graph = helper.make_graph(
            [helper.make_node("Identity", ["S" if used else "X"], ["Y"])],
            "sparse",
            []
            if nested
            else [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
            sparse_initializer=[sparse],
        )
        if nested:
            graph = helper.make_graph(
                [
                    helper.make_node(
                        "If", ["cond"], ["Z"], then_branch=graph, else_branch=graph
                    )
                ],
                "outer",
                [
                    helper.make_tensor_value_info("X", TensorProto.FLOAT, [1]),
                    helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                ],
                [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1])],
            )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        checker.check_model(model)
        with pytest.raises(
            onnx.version_converter.ConvertError,
            match="Sparse initializers are not supported",
        ):
            onnx.version_converter.convert_version(model, 14)
