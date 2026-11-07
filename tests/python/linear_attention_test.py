# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import patch

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import onnx
from onnx import helper, inliner
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRunExpand
from onnx.reference.ops.op_linear_attention import (
    LinearAttention as ReferenceLinearAttention,
)


class LinearAttention(OpRunExpand):
    op_domain = ""


def _function_body(node, types, version=29):
    return onnx.defs.get_schema(
        "LinearAttention", version
    ).get_context_dependent_function(
        node.SerializeToString(), [t.SerializeToString() for t in types]
    )


@pytest.mark.parametrize("trajectory", [False, True])
def test_linked_value_heads_require_binding(trajectory):
    feeds = {
        "query": np.array([[[1, 2]]], dtype=np.float32),
        "key": np.array([[[1, 2]]], dtype=np.float32),
        "value": np.array([[[3, 5]]], dtype=np.float32),
    }
    attrs = {
        "q_num_heads": 1,
        "kv_num_heads": 1,
        "v_num_heads": 2,
        "update_rule": "linear",
        "scale": 1.0,
    }
    model = _model(feeds, attrs, trajectory=trajectory)
    node = model.graph.node[0]
    linked = next(attr for attr in node.attribute if attr.name == "v_num_heads")
    linked.ClearField("i")
    linked.ref_attr_name = "heads"
    types = [value.type for value in model.graph.input]
    assert not _function_body(node, types)

    function = helper.make_function(
        "local",
        "linked_attention",
        list(feeds),
        list(node.output),
        [node],
        [helper.make_opsetid("", 29)],
        attributes=["heads"],
    )
    call = helper.make_node(
        function.name, list(feeds), list(node.output), domain=function.domain, heads=2
    )
    model.graph.node[0].CopyFrom(call)
    model.functions.append(function)
    model.opset_import.append(helper.make_opsetid(function.domain, 1))
    selected = inliner.inline_selected_functions(
        model, [("", "LinearAttention")], inline_schema_functions=True
    )
    assert selected.functions[0].node[0].op_type == "LinearAttention"
    expected_state = np.array([[[[3], [6]], [[5], [10]]]], dtype=np.float32)
    bound = inliner.inline_local_functions(selected)
    expanded = inliner.inline_selected_functions(
        bound, [("", "LinearAttention")], inline_schema_functions=True
    )
    assert all(node.op_type != "LinearAttention" for node in expanded.graph.node)
    onnx.checker.check_model(expanded, full_check=True)
    for candidate in (selected, bound, expanded):
        actual = ReferenceEvaluator(candidate).run(None, feeds)
        assert actual[1].shape == (1, 2, 2, 1)
        assert_array_equal(actual[1], expected_state)
        if trajectory:
            assert_array_equal(actual[2][:, 0], expected_state)


@pytest.mark.parametrize("version", [27, 29])
def test_value_head_function_representation_is_bounded(version):
    sizes = []
    counts = []
    for heads in (2, 32, 1024, 4096):
        attrs = {
            "q_num_heads": 1,
            "kv_num_heads": 1,
            "update_rule": "linear",
            "scale": 1.0,
        }
        if version == 29:
            attrs["v_num_heads"] = heads
        node = helper.make_node("LinearAttention", ["Q", "K", "V"], ["Y", "S"], **attrs)
        types = [
            helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape)
            for shape in ((0, 0, 1), (0, 0, 1), (0, 0, heads))
        ]
        body = _function_body(node, types, version)
        function = onnx.FunctionProto.FromString(body)
        sizes.append(len(body))
        counts.append(len(function.node))
        assert all(
            len(attr.t.int64_data) <= 5
            for op in function.node
            for attr in op.attribute
            if attr.HasField("t")
        )
        output, state = ReferenceEvaluator(function).run(
            None,
            {
                "query": np.empty((0, 0, 1), dtype=np.float32),
                "key": np.empty((0, 0, 1), dtype=np.float32),
                "value": np.empty((0, 0, heads), dtype=np.float32),
            },
            attributes={},
        )
        assert output.shape == (0, 0, heads)
        assert state.shape == (
            0,
            heads if version == 29 else 1,
            1,
            1 if version == 29 else heads,
        )
    assert len(set(counts)) == 1
    assert max(sizes) - min(sizes) <= 64
    assert max(sizes) < 10_000


@pytest.mark.parametrize("trajectory", [False, True])
@pytest.mark.parametrize("length", [0, 2])
@pytest.mark.parametrize("past", [False, True])
@pytest.mark.parametrize("rule", ["linear", "gated", "delta", "gated_delta"])
def test_inlined_linear_attention_captures(trajectory, length, past, rule):
    feeds, attrs = _case(3, length, rule, np.float32, past)
    model = _model(feeds, attrs, trajectory=trajectory, empty_slots=True)
    expected = ReferenceEvaluator(model).run(None, feeds)
    expanded = inliner.inline_selected_functions(
        model, [("", "LinearAttention")], inline_schema_functions=True
    )
    assert all(node.op_type != "LinearAttention" for node in expanded.graph.node)
    onnx.checker.check_model(expanded, full_check=True)
    actual = ReferenceEvaluator(expanded).run(None, feeds)
    for got, want in zip(actual, expected, strict=True):
        assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def _model(feeds, attrs, version=29, *, trajectory=False, empty_slots=False):
    names = ["query", "key", "value", "past_state", "decay", "beta"]
    inputs = [name if name in feeds else "" for name in names]
    while not empty_slots and not inputs[-1]:
        inputs.pop()
    outputs = ["output", "present_state"]
    if trajectory or empty_slots:
        outputs.append("states" if trajectory else "")
    node = helper.make_node("LinearAttention", inputs, outputs, **attrs)
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
    graph_outputs = [
        helper.make_tensor_value_info("output", activation_type, [None] * 3),
        helper.make_tensor_value_info("present_state", state_type, [None] * 4),
    ]
    if trajectory:
        graph_outputs.append(
            helper.make_tensor_value_info("states", state_type, [None] * 5)
        )
    graph = helper.make_graph([node], "linear_attention", graph_inputs, graph_outputs)
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
            => (float[1,1,4] Y, float[1,4,1,1] Next, float[1,1,4,1,1] States) {
            Y, Next, States = LinearAttention <
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
        output, state, states = ReferenceEvaluator(model, new_ops=new_ops).run(
            None, feeds
        )
        assert_allclose(output, [[[2, 3, 60, 100]]])
        assert_allclose(state.reshape(4), [2, 3, 6, 10])
        assert_array_equal(states[:, 0], state)


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


def _trajectory_case(heads, length, rule, dtype, past, key_decay, scalar_beta):
    q_heads, k_heads, v_heads = heads
    state_heads = v_heads or k_heads
    rng = np.random.default_rng(42)
    batch, dk, dv = 2, 3, 5
    key = rng.normal(size=(batch, length, k_heads, dk)).astype(np.float32)
    key /= np.maximum(np.linalg.norm(key, axis=-1, keepdims=True), 1e-6)
    feeds = {
        "query": rng.normal(size=(batch, length, q_heads * dk)).astype(dtype),
        "key": key.reshape(batch, length, k_heads * dk).astype(dtype),
        "value": rng.normal(size=(batch, length, state_heads * dv)).astype(dtype),
    }
    if past:
        feeds["past_state"] = rng.normal(size=(batch, state_heads, dk, dv)).astype(
            np.float32
        )
    if rule in ("gated", "gated_delta"):
        feeds["decay"] = -rng.uniform(
            0.1, 0.9, (batch, length, state_heads * (dk if key_decay else 1))
        ).astype(dtype)
    if rule in ("delta", "gated_delta"):
        feeds["beta"] = rng.uniform(
            0.1, 0.9, (batch, length, 1 if scalar_beta else state_heads)
        ).astype(dtype)
    return feeds, {
        "q_num_heads": q_heads,
        "kv_num_heads": k_heads,
        "v_num_heads": v_heads,
        "update_rule": rule,
        "scale": 0.7,
    }


def _trajectory_expected(feeds, attrs):
    q_heads, k_heads = attrs["q_num_heads"], attrs["kv_num_heads"]
    state_heads = attrs["v_num_heads"] or k_heads
    output_heads = attrs["v_num_heads"] or q_heads
    batch, length, _ = feeds["query"].shape
    dk = feeds["key"].shape[-1] // k_heads
    dv = feeds["value"].shape[-1] // state_heads
    query = feeds["query"].astype(np.float32).reshape(batch, length, q_heads, dk)
    key = feeds["key"].astype(np.float32).reshape(batch, length, k_heads, dk)
    value = feeds["value"].astype(np.float32).reshape(batch, length, state_heads, dv)
    state_dtype = feeds.get("past_state", feeds["query"]).dtype
    state = feeds.get(
        "past_state", np.zeros((batch, state_heads, dk, dv), dtype=np.float32)
    ).astype(np.float32)
    states = np.empty((batch, length, state_heads, dk, dv), dtype=state_dtype)
    output = np.empty((batch, length, output_heads, dv), dtype=feeds["query"].dtype)
    for t in range(length):
        for b in range(batch):
            for h in range(state_heads):
                kh = h // (state_heads // k_heads)
                k = key[b, t, kh]
                if "decay" in feeds:
                    decay = (
                        feeds["decay"][b, t]
                        .astype(np.float32)
                        .reshape(state_heads, -1)[h]
                    )
                    state[b, h] *= np.exp(decay)[:, None]
                update = value[b, t, h]
                if "beta" in feeds:
                    beta = feeds["beta"][b, t].astype(np.float32)
                    update = beta[0 if beta.size == 1 else h] * (
                        update - k @ state[b, h]
                    )
                state[b, h] += np.outer(k, update)
            for h in range(output_heads):
                sh = h if attrs["v_num_heads"] else h // (q_heads // k_heads)
                qh = h // (state_heads // k_heads) if attrs["v_num_heads"] else h
                output[b, t, h] = attrs["scale"] * (query[b, t, qh] @ state[b, sh])
        states[:, t] = state
    return (
        output.reshape(batch, length, output_heads * dv),
        state.astype(state_dtype),
        states,
    )


@pytest.mark.parametrize(
    "heads",
    [
        (2, 2, 0),
        (4, 2, 0),
        (4, 1, 0),
        (2, 2, 2),
        (2, 2, 4),
        (2, 2, 6),
        (2, 2, 16),
        (1, 1, 16),
    ],
)
@pytest.mark.parametrize("length", [0, 1, 4])
@pytest.mark.parametrize("rule", ["linear", "gated", "delta", "gated_delta"])
@pytest.mark.parametrize("dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("past", [False, True])
def test_state_trajectory(heads, length, rule, dtype, past):
    feeds, attrs = _trajectory_case(heads, length, rule, dtype, past, True, True)
    expected = _trajectory_expected(feeds, attrs)
    model = _model(feeds, attrs, trajectory=True, empty_slots=True)
    original = {name: value.copy() for name, value in feeds.items()}
    for new_ops in (None, [LinearAttention]):
        actual = ReferenceEvaluator(model, new_ops=new_ops).run(None, feeds)
        baseline = ReferenceEvaluator(_model(feeds, attrs), new_ops=new_ops).run(
            None, feeds
        )
        for got, want in zip(actual, expected, strict=True):
            assert got.shape == want.shape
            assert got.dtype == want.dtype
            tolerance = max(3e-5, 2 * ml_dtypes.finfo(got.dtype).eps)
            assert_allclose(
                got.astype(np.float32),
                want.astype(np.float32),
                rtol=tolerance,
                atol=2e-6,
            )
        for got, want in zip(actual[:2], baseline, strict=True):
            assert_array_equal(got, want)
        if length:
            assert_array_equal(actual[2][:, -1], actual[1])
        for name, value in feeds.items():
            assert_array_equal(value, original[name])


@pytest.mark.parametrize("state_dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize(("key_decay", "scalar_beta"), [(False, False), (True, True)])
@pytest.mark.parametrize("expanded", [False, True])
def test_state_trajectory_streaming_and_prefix_selection(
    state_dtype, key_decay, scalar_beta, expanded
):
    feeds, attrs = _trajectory_case(
        (2, 2, 6), 5, "gated_delta", np.float16, True, key_decay, scalar_beta
    )
    feeds["past_state"] = feeds["past_state"].astype(state_dtype)
    model = _model(feeds, attrs, trajectory=True)
    new_ops = [LinearAttention] if expanded else None
    actual = ReferenceEvaluator(model, new_ops=new_ops).run(None, feeds)
    expected = _trajectory_expected(feeds, attrs)
    assert_array_equal(actual[2][:, -1], actual[1])
    for got, want in zip(actual, expected, strict=True):
        assert_allclose(
            got.astype(np.float32),
            want.astype(np.float32),
            rtol=max(3e-5, 2 * ml_dtypes.finfo(got.dtype).eps),
            atol=2e-6,
        )
    # Gather is caller-owned selection; accepting no tokens keeps the initial state.
    model.graph.input.append(
        helper.make_tensor_value_info("accepted", onnx.TensorProto.INT64, [])
    )
    model.graph.node.append(
        helper.make_node("Gather", ["states", "accepted"], ["selected"], axis=1)
    )
    model.graph.output.append(
        helper.make_tensor_value_info(
            "selected",
            helper.np_dtype_to_tensor_dtype(np.dtype(state_dtype)),
            [2, 6, 3, 5],
        )
    )
    onnx.checker.check_model(model, full_check=True)
    for accepted in (0, 1, 3, 5):
        selected = (
            feeds["past_state"]
            if accepted == 0
            else ReferenceEvaluator(model, new_ops=new_ops).run(
                ["selected"],
                {**feeds, "accepted": np.array(accepted - 1, dtype=np.int64)},
            )[0]
        )
        assert_array_equal(
            selected,
            feeds["past_state"] if accepted == 0 else actual[2][:, accepted - 1],
        )
        if state_dtype != np.float32:
            continue
        prefix = {
            name: value if name == "past_state" else value[:, :accepted]
            for name, value in feeds.items()
        }
        prefix_state = ReferenceEvaluator(_model(prefix, attrs), new_ops=new_ops).run(
            None, prefix
        )[1]
        assert_allclose(selected, prefix_state, rtol=1e-6, atol=1e-6)
        suffix = {
            name: value[:, accepted:]
            for name, value in feeds.items()
            if name != "past_state"
        }
        suffix["past_state"] = selected
        suffix_results = ReferenceEvaluator(
            _model(suffix, attrs, trajectory=True), new_ops=new_ops
        ).run(None, suffix)
        assert_array_equal(suffix_results[0], actual[0][:, accepted:])
        assert_allclose(suffix_results[1], actual[1], rtol=1e-6, atol=1e-6)
        assert_allclose(
            suffix_results[2], actual[2][:, accepted:], rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize("empty_slots", [False, True])
@pytest.mark.parametrize("rule", ["linear", "gated", "delta", "gated_delta"])
def test_omitted_trajectory_has_no_snapshot_storage(empty_slots, rule):
    feeds, attrs = _trajectory_case((2, 2, 4), 3, rule, np.float32, False, False, False)
    model = _model(feeds, attrs, empty_slots=empty_slots)
    evaluator = ReferenceEvaluator(model)
    with patch(
        "onnx.reference.ops.op_linear_attention.np.empty", wraps=np.empty
    ) as allocate:
        actual = evaluator.run(None, feeds)
    assert not allocate.called
    schema = onnx.defs.get_schema("LinearAttention", 29)
    types = [
        helper.make_tensor_type_proto(
            helper.np_dtype_to_tensor_dtype(feeds[name].dtype), feeds[name].shape
        )
        if name
        else onnx.TypeProto()
        for name in model.graph.node[0].input
    ]
    function = onnx.FunctionProto.FromString(
        schema.get_context_dependent_function(
            model.graph.node[0].SerializeToString(),
            [t.SerializeToString() for t in types],
        )
    )
    assert list(function.output) == ["output", "present_state"]
    conditional = next(node for node in function.node if node.op_type == "If")
    for branch in conditional.attribute:
        assert len(branch.g.output) == 2
        for node in branch.g.node:
            if node.op_type == "Scan":
                assert len(node.output) == 2
                body = next(attr.g for attr in node.attribute if attr.name == "body")
                assert len(body.output) == 2
    expanded = ReferenceEvaluator(model, new_ops=[LinearAttention]).run(None, feeds)
    for got, want in zip(expanded, actual, strict=True):
        assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_state_trajectory_is_opset29_only():
    for version, count in [(27, 2), (28, 2), (29, 3)]:
        schema = onnx.defs.get_schema("LinearAttention", version)
        assert len(schema.outputs) == count
        if version == 29:
            assert schema.outputs[2].name == "states"
            assert (
                schema.outputs[2].option
                == onnx.defs.OpSchema.FormalParameterOption.Optional
            )
            assert schema.outputs[2].type_str == "S"
