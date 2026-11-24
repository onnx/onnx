# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import onnx
import onnx.reference
from onnx import TensorProto, checker, helper, shape_inference
from onnx.defs import ONNX_EXPERIMENTAL_STATE_DOMAIN


def test_reference_schema_lookup_allows_cross_domain_names() -> None:
    assert (
        onnx.reference.op_run._schemas[
            (ONNX_EXPERIMENTAL_STATE_DOMAIN, "CausalConvWithState")
        ].domain
        == ONNX_EXPERIMENTAL_STATE_DOMAIN
    )
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
    assert [
        dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim
    ] == [
        2,
        4,
        8,
    ]
    state_type = inferred.graph.output[1].type.opaque_type
    assert state_type.domain == ONNX_EXPERIMENTAL_STATE_DOMAIN
    assert state_type.name == "CausalConvState"


def test_attention_with_cache_packed_inference() -> None:
    node = helper.make_node(
        "AttentionWithCache",
        [
            "Q",
            "K",
            "V",
            "",
            "",
            "q_positions",
            "kv_positions",
            "",
            "q_offsets",
            "kv_offsets",
            "visibility_rows",
            "visibility_blocks",
            "partial_indices",
            "partial_masks",
            "attn_bias",
        ],
        ["Y", "cache"],
        domain=ONNX_EXPERIMENTAL_STATE_DOMAIN,
        q_num_heads=4,
        kv_num_heads=2,
        visibility_q_block_size=2,
        visibility_k_block_size=65,
        visibility_num_heads=4,
    )
    initializers = [
        helper.make_tensor("q_offsets", TensorProto.INT32, [3], [0, 3, 3]),
        helper.make_tensor("kv_offsets", TensorProto.INT32, [3], [0, 2, 3]),
        helper.make_tensor(
            "visibility_rows",
            TensorProto.INT32,
            [9],
            [0, 2, 2, 2, 2, 2, 2, 2, 2],
        ),
        helper.make_tensor("visibility_blocks", TensorProto.INT32, [2], [0, 1]),
        helper.make_tensor("partial_indices", TensorProto.INT32, [1], [1]),
    ]
    graph = helper.make_graph(
        [node],
        "experimental-packed-attention",
        [
            helper.make_tensor_value_info("Q", TensorProto.FLOAT, [3, 16]),
            helper.make_tensor_value_info("K", TensorProto.FLOAT, [3, 8]),
            helper.make_tensor_value_info("V", TensorProto.FLOAT, [3, 12]),
            helper.make_tensor_value_info("q_positions", TensorProto.INT32, [3]),
            helper.make_tensor_value_info("kv_positions", TensorProto.INT32, [3]),
            helper.make_tensor_value_info(
                "partial_masks", TensorProto.UINT64, [1, 2, 2]
            ),
            helper.make_tensor_value_info("attn_bias", TensorProto.FLOAT, [2, 2, 65]),
        ],
        [
            helper.make_empty_tensor_value_info("Y"),
            helper.make_empty_tensor_value_info("cache"),
        ],
        initializer=initializers,
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
    assert [
        dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim
    ] == [
        3,
        24,
    ]


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    [
        (
            ["Q", "K", "V"],
            ["Y", "cache"],
            "require q_sequence_offsets and kv_sequence_offsets",
        ),
        (
            ["Q", "K", "V", "", "", "", "", "", "q_offsets", "kv_offsets"],
            ["Y", "cache", "scores"],
            "does not define the dense qk_matmul_output",
        ),
    ],
)
def test_attention_with_cache_rejects_invalid_packed_forms(
    inputs: list[str], outputs: list[str], message: str
) -> None:
    node = helper.make_node(
        "AttentionWithCache",
        inputs,
        outputs,
        domain=ONNX_EXPERIMENTAL_STATE_DOMAIN,
        q_num_heads=2,
        kv_num_heads=2,
    )
    graph_inputs = [
        helper.make_tensor_value_info("Q", TensorProto.FLOAT, [2, 8]),
        helper.make_tensor_value_info("K", TensorProto.FLOAT, [2, 8]),
        helper.make_tensor_value_info("V", TensorProto.FLOAT, [2, 8]),
    ]
    initializers = []
    if "q_offsets" in inputs:
        initializers.extend(
            [
                helper.make_tensor("q_offsets", TensorProto.INT64, [2], [0, 2]),
                helper.make_tensor("kv_offsets", TensorProto.INT64, [2], [0, 2]),
            ]
        )
    graph = helper.make_graph(
        [node],
        "invalid-packed-attention",
        graph_inputs,
        [helper.make_empty_tensor_value_info(name) for name in outputs],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", onnx.defs.onnx_opset_version()),
            helper.make_opsetid(ONNX_EXPERIMENTAL_STATE_DOMAIN, 1),
        ],
    )

    with pytest.raises(shape_inference.InferenceError, match=message):
        shape_inference.infer_shapes(model, strict_mode=True)


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
