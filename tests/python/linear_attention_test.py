# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import onnx
from onnx import helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRunExpand
from onnx.reference.ops.op_linear_attention import (
    LinearAttention as ReferenceLinearAttention,
)


class LinearAttention(OpRunExpand):
    op_domain = ""


def _model(feeds, attrs, version=29):
    names = ["query", "key", "value", "past_state", "decay", "beta"]
    inputs = [name if name in feeds else "" for name in names]
    while not inputs[-1]:
        inputs.pop()
    node = helper.make_node(
        "LinearAttention", inputs, ["output", "present_state"], **attrs
    )
    graph_inputs = [
        helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
        )
        for name, value in feeds.items()
    ]
    activation_type = helper.np_dtype_to_tensor_dtype(feeds["query"].dtype)
    state_type = helper.np_dtype_to_tensor_dtype(
        feeds.get("past_state", feeds["query"]).dtype
    )
    graph = helper.make_graph(
        [node],
        "linear_attention",
        graph_inputs,
        [
            helper.make_tensor_value_info("output", activation_type, [None] * 3),
            helper.make_tensor_value_info("present_state", state_type, [None] * 4),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", version)])
    onnx.checker.check_model(model, full_check=True)
    return model


def _case(factor, length, rule, dtype, past, key_decay=False, scalar_beta=False):
    rng = np.random.default_rng(42)
    batch, heads, dk, dv = 2, 2, 3, 5
    value_heads = heads * factor
    feeds = {
        "query": rng.normal(size=(batch, length, heads * dk)).astype(dtype),
        "key": rng.normal(size=(batch, length, heads * dk)).astype(dtype),
        "value": rng.normal(size=(batch, length, value_heads * dv)).astype(dtype),
    }
    key = feeds["key"].astype(np.float32).reshape(batch, length, heads, dk)
    key /= np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1e-6)
    feeds["key"] = key.reshape(batch, length, heads * dk).astype(dtype)
    if past:
        feeds["past_state"] = rng.normal(size=(batch, value_heads, dk, dv)).astype(
            np.float32
        )
    if rule in ("gated", "gated_delta"):
        width = value_heads * (dk if key_decay else 1)
        feeds["decay"] = -rng.uniform(0.1, 0.9, (batch, length, width)).astype(dtype)
    if rule in ("delta", "gated_delta"):
        width = 1 if scalar_beta else value_heads
        feeds["beta"] = rng.uniform(0.1, 0.9, (batch, length, width)).astype(dtype)
    attrs = {
        "q_num_heads": heads,
        "kv_num_heads": heads,
        "v_num_heads": value_heads,
        "update_rule": rule,
    }
    return feeds, attrs


def _expanded_legacy(feeds, attrs):
    factor = attrs["v_num_heads"] // attrs["kv_num_heads"]
    expanded = feeds.copy()
    for name in ("query", "key"):
        x = feeds[name]
        batch, length, width = x.shape
        heads = attrs["kv_num_heads"]
        expanded[name] = np.repeat(
            x.reshape(batch, length, heads, width // heads), factor, axis=2
        ).reshape(batch, length, width * factor)
    legacy_attrs = {
        **attrs,
        "q_num_heads": attrs["v_num_heads"],
        "kv_num_heads": attrs["v_num_heads"],
    }
    del legacy_attrs["v_num_heads"]
    return ReferenceEvaluator(_model(expanded, legacy_attrs, 27)).run(None, expanded)


@pytest.mark.parametrize("factor", [2, 3, 8])
@pytest.mark.parametrize("length", [0, 1, 5])
@pytest.mark.parametrize("rule", ["linear", "gated", "delta", "gated_delta"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("past", [False, True])
def test_compact_heads_reference_and_function(factor, length, rule, dtype, past):
    feeds, attrs = _case(factor, length, rule, dtype, past)
    model = _model(feeds, attrs)
    expected = _expanded_legacy(feeds, attrs)
    for new_ops in (None, [LinearAttention]):
        actual = ReferenceEvaluator(model, new_ops=new_ops).run(None, feeds)
        for got, want in zip(actual, expected, strict=True):
            assert got.shape == want.shape
            assert got.dtype == want.dtype
            tolerance = max(2e-5, 2 * ml_dtypes.finfo(got.dtype).eps)
            assert_allclose(
                got.astype(np.float32),
                want.astype(np.float32),
                rtol=tolerance,
                atol=2e-6,
            )
        if length == 0:
            assert_array_equal(
                actual[1], feeds.get("past_state", np.zeros_like(actual[1]))
            )


@pytest.mark.parametrize("factor", [2, 3, 8])
@pytest.mark.parametrize("key_decay", [False, True])
@pytest.mark.parametrize("scalar_beta", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
def test_compact_heads_streaming(factor, key_decay, scalar_beta, expanded):
    feeds, attrs = _case(
        factor, 5, "gated_delta", np.float16, True, key_decay, scalar_beta
    )
    attrs["scale"] = 0.7
    new_ops = [LinearAttention] if expanded else None
    expected = ReferenceEvaluator(_model(feeds, attrs), new_ops=new_ops).run(
        None, feeds
    )
    state = feeds["past_state"]
    chunks = []
    for start, end in [(0, 2), (2, 2), (2, 3), (3, 5)]:
        chunk = {
            name: value[:, start:end]
            for name, value in feeds.items()
            if name != "past_state"
        }
        chunk["past_state"] = state
        chunk_attrs = {**attrs, "chunk_size": 1}
        output, state = ReferenceEvaluator(
            _model(chunk, chunk_attrs), new_ops=new_ops
        ).run(None, chunk)
        chunks.append(output)
    assert_array_equal(np.concatenate(chunks, axis=1), expected[0])
    assert_allclose(state, expected[1], rtol=1e-6, atol=1e-6)
    assert state.dtype == np.float32


@pytest.mark.parametrize("version", [27, 29])
@pytest.mark.parametrize("query_heads", [2, 4, 8])
def test_legacy_query_gqa_function(version, query_heads):
    feeds, attrs = _case(1, 3, "gated_delta", np.float32, True, True, True)
    del attrs["v_num_heads"]
    attrs["q_num_heads"] = query_heads
    feeds["query"] = np.tile(feeds["query"], (1, 1, query_heads // 2))
    model = _model(feeds, attrs, version)
    expected = ReferenceEvaluator(model).run(None, feeds)
    actual = ReferenceEvaluator(model, new_ops=[LinearAttention]).run(None, feeds)
    for got, want in zip(actual, expected, strict=True):
        assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_compact_head_order_and_decayed_delta():
    model = onnx.parser.parse_model(
        """
        <ir_version: 13, opset_import: ["" : 29]>
        g (float[1,1,2] Q, float[1,1,2] K, float[1,1,4] V,
           float[1,4,1,1] S, float[1,1,4] D, float[1,1,4] Beta)
            => (float[1,1,4] Y, float[1,4,1,1] Next) {
            Y, Next = LinearAttention <
                q_num_heads=2, kv_num_heads=2, v_num_heads=4, scale=1.0
            > (Q, K, V, S, D, Beta)
        }
        """
    )
    feeds = {
        "Q": np.array([[[1, 10]]], dtype=np.float32),
        "K": np.array([[[1, 2]]], dtype=np.float32),
        "V": np.array([[[3, 5, 7, 11]]], dtype=np.float32),
        "S": np.full((1, 4, 1, 1), 2, dtype=np.float32),
        "D": np.full((1, 1, 4), np.log(0.5), dtype=np.float32),
        "Beta": np.full((1, 1, 4), 0.5, dtype=np.float32),
    }
    # Decayed state is 1; correction uses [1,1,2,2], then writes and reads.
    for new_ops in (None, [LinearAttention]):
        output, state = ReferenceEvaluator(model, new_ops=new_ops).run(None, feeds)
        assert_allclose(output, [[[2, 3, 60, 100]]])
        assert_allclose(state.reshape(4), [2, 3, 6, 10])


@pytest.mark.parametrize(("version", "has_value_heads"), [(27, False), (29, True)])
def test_legacy_schema_has_no_value_head_attribute(version, has_value_heads):
    schema = onnx.defs.get_schema("LinearAttention", version)
    assert ("v_num_heads" in schema.attributes) == has_value_heads
    assert schema.since_version == version


@pytest.mark.parametrize("query_heads", [2, 4, 8])
def test_default_value_heads_preserve_query_gqa(query_heads):
    feeds, attrs = _case(1, 3, "linear", np.float32, False)
    del attrs["v_num_heads"]
    attrs["q_num_heads"] = query_heads
    feeds["query"] = np.tile(feeds["query"], (1, 1, query_heads // 2))
    legacy = ReferenceEvaluator(_model(feeds, attrs, 27)).run(None, feeds)
    for default_attrs in (attrs, {**attrs, "v_num_heads": 0}):
        for new_ops in (None, [LinearAttention]):
            actual = ReferenceEvaluator(
                _model(feeds, default_attrs), new_ops=new_ops
            ).run(None, feeds)
            for got, want in zip(actual, legacy, strict=True):
                assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "attrs_override",
    [
        {"v_num_heads": -1},
        {"v_num_heads": 3},
        {"q_num_heads": 4},
    ],
)
def test_compact_heads_invalid_runtime_attributes(attrs_override):
    feeds, attrs = _case(2, 1, "gated_delta", np.float32, True)
    op = ReferenceLinearAttention.__new__(ReferenceLinearAttention)
    with pytest.raises(ValueError, match="v_num_heads"):
        op._run(**feeds, **{**attrs, **attrs_override})


@pytest.mark.parametrize("name", ["value", "past_state", "decay", "beta"])
def test_compact_heads_invalid_runtime_shapes(name):
    feeds, attrs = _case(2, 1, "gated_delta", np.float32, True)
    value = feeds[name]
    feeds[name] = value[:, :2] if name == "past_state" else value[..., :-1]
    op = ReferenceLinearAttention.__new__(ReferenceLinearAttention)
    with pytest.raises(
        ValueError, match=r"num_heads|state head count|past_state shape"
    ):
        op._run(**feeds, **attrs)


@pytest.mark.parametrize("version", [27, 28])
@pytest.mark.parametrize("length", [0, 1, 3])
@pytest.mark.parametrize("past", [False, True])
@pytest.mark.parametrize("query_heads", [1, 2])
@pytest.mark.parametrize("empty_dimension", ["key", "value"])
@pytest.mark.parametrize("rule", ["linear", "gated_delta"])
def test_legacy_zero_width_heads(
    version, length, past, query_heads, empty_dimension, rule
):
    dk = 0 if empty_dimension == "key" else 1
    dv = 0 if empty_dimension == "value" else 1
    feeds = {
        "query": np.ones((1, length, query_heads * dk), dtype=np.float32),
        "key": np.ones((1, length, dk), dtype=np.float32),
        "value": np.ones((1, length, dv), dtype=np.float32),
    }
    if past:
        feeds["past_state"] = np.empty((1, 1, dk, dv), dtype=np.float32)
    if rule == "gated_delta":
        feeds["decay"] = np.zeros((1, length, dk), dtype=np.float32)
        feeds["beta"] = np.ones((1, length, 1), dtype=np.float32)
    attrs = {
        "q_num_heads": query_heads,
        "kv_num_heads": 1,
        "update_rule": rule,
        "scale": 1.0,
    }
    model = _model(feeds, attrs, version)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    expected_shapes = [(1, length, query_heads * dv), (1, 1, dk, dv)]
    assert [
        tuple(dim.dim_value for dim in output.type.tensor_type.shape.dim)
        for output in inferred.graph.output
    ] == expected_shapes
    for new_ops in (None, [LinearAttention]):
        outputs = ReferenceEvaluator(model, new_ops=new_ops).run(None, feeds)
        for output, shape in zip(outputs, expected_shapes, strict=True):
            assert output.shape == shape
            assert_array_equal(output, np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("value_heads", [None, 0, 2])
@pytest.mark.parametrize("empty_dimension", ["key", "value"])
def test_opset29_rejects_zero_width_heads(value_heads, empty_dimension):
    dk = 0 if empty_dimension == "key" else 1
    dv = 0 if empty_dimension == "value" else 1
    heads = value_heads or 1
    feeds = {
        "query": np.ones((1, 1, dk), dtype=np.float32),
        "key": np.ones((1, 1, dk), dtype=np.float32),
        "value": np.ones((1, 1, heads * dv), dtype=np.float32),
    }
    attrs = {
        "q_num_heads": 1,
        "kv_num_heads": 1,
        "update_rule": "linear",
        "scale": 1.0,
    }
    model = _model(feeds, attrs, 27)
    model.opset_import[0].version = 29
    if value_heads is not None:
        model.graph.node[0].attribute.append(
            helper.make_attribute("v_num_heads", value_heads)
        )
    with pytest.raises(onnx.shape_inference.InferenceError, match="must be positive"):
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
    with pytest.raises(ValueError, match="must be positive"):
        ReferenceEvaluator(model).run(None, feeds)
