# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import onnx
from onnx import TensorProto, helper
from onnx.defs import ONNX_DOMAIN


@pytest.mark.parametrize("version", [2, 11, 13])
def test_split_requires_output(version: int) -> None:
    graph = helper.make_graph(
        [helper.make_node("Split", ["x"], [])],
        "test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, (2,))],
        [],
    )
    model = helper.make_model(
        graph,
        producer_name="onnx-test",
        opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
    )
    with pytest.raises(
        onnx.shape_inference.InferenceError, match="at least one output"
    ):
        onnx.shape_inference.infer_shapes(
            model,
            check_type=True,
            strict_mode=True,
        )
