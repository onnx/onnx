# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import onnx
import onnx.reference
from onnx import TensorProto, checker, helper, shape_inference
from onnx.defs import ONNX_EXPERIMENTAL_STATE_DOMAIN


def test_reference_schema_lookup_allows_cross_domain_names() -> None:
    assert onnx.reference.op_run._schemas[
        (ONNX_EXPERIMENTAL_STATE_DOMAIN, "CausalConvWithState")
    ].domain == ONNX_EXPERIMENTAL_STATE_DOMAIN
    assert onnx.reference.op_run._schemas[("", "CausalConvWithState")].domain == ""


@pytest.mark.parametrize(
    ("op_type", "state_name"),
    [
        ("AttentionWithCache", "KVCache"),
        ("LinearAttentionWithState", "LinearAttentionState"),
        ("CausalConvWithState", "CausalConvState"),
    ],
)
def test_experimental_state_schema_uses_nominal_opaque_state(
    op_type: str, state_name: str
) -> None:
    schema = onnx.defs.get_schema(op_type, 1, ONNX_EXPERIMENTAL_STATE_DOMAIN)
    state_output = schema.outputs[1]
    assert state_output.type_str == "S"
    assert state_output.types == {
        f"opaque({ONNX_EXPERIMENTAL_STATE_DOMAIN},{state_name})"
    }
    assert schema.support_level == onnx.defs.OpSchema.SupportType.EXPERIMENTAL


def test_causal_conv_with_state_inference() -> None:
    node = helper.make_node(
        "CausalConvWithState",
        ["X", "W", "", "", "lengths"],
        ["Y", "state"],
        domain=ONNX_EXPERIMENTAL_STATE_DOMAIN,
        dilation=2,
    )
    graph = helper.make_graph(
        [node],
        "experimental-causal-conv",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4, 8]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [4, 1, 3]),
            helper.make_tensor_value_info("lengths", TensorProto.INT64, [2]),
        ],
        [
            helper.make_empty_tensor_value_info("Y"),
            helper.make_empty_tensor_value_info("state"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", onnx.defs.onnx_opset_version()),
            helper.make_opsetid(ONNX_EXPERIMENTAL_STATE_DOMAIN, 1),
        ],
    )

    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    checker.check_model(inferred, full_check=True)
    assert [dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim] == [
        2,
        4,
        8,
    ]
    state_type = inferred.graph.output[1].type.opaque_type
    assert state_type.domain == ONNX_EXPERIMENTAL_STATE_DOMAIN
    assert state_type.name == "CausalConvState"


def test_experimental_state_rejects_invalid_lossy_flag() -> None:
    node = helper.make_node(
        "CausalConvWithState",
        ["X", "W"],
        ["Y", "state"],
        domain=ONNX_EXPERIMENTAL_STATE_DOMAIN,
        allow_lossy_state=2,
    )
    graph = helper.make_graph(
        [node],
        "invalid-lossy-state",
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [2, 1, 2]),
        ],
        [
            helper.make_empty_tensor_value_info("Y"),
            helper.make_empty_tensor_value_info("state"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", onnx.defs.onnx_opset_version()),
            helper.make_opsetid(ONNX_EXPERIMENTAL_STATE_DOMAIN, 1),
        ],
    )

    with pytest.raises(shape_inference.InferenceError, match="must be 0 or 1"):
        shape_inference.infer_shapes(model, strict_mode=True)
