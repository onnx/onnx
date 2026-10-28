# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

import onnx
from onnx.backend.test.case.node import function_testcase_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops._op_list import load_op
from onnx.reference.ops.op_attention import _compute_attention


def test_attention_reference_uses_versioned_implementations() -> None:
    assert load_op("", "Attention", 22).__name__ == "Attention_1"
    assert load_op("", "Attention", 23).__name__ == "Attention_23"
    assert load_op("", "Attention", 24).__name__ == "Attention_24"
    assert load_op("", "Attention", 25).__name__ == "Attention_25"
    assert load_op("", "Attention", 28).__name__ == "Attention_25"
    assert load_op("", "Attention", 29).__name__ == "Attention_29"


def test_attention_external_cache_rank3_mask_is_head_broadcast() -> None:
    batch_size, num_heads, q_len, kv_len, head_size = 2, 3, 2, 4, 2
    q = np.zeros((batch_size, num_heads, q_len, head_size), dtype=np.float32)
    k = np.zeros((batch_size, num_heads, kv_len, head_size), dtype=np.float32)
    v = np.arange(
        batch_size * num_heads * kv_len * head_size, dtype=np.float32
    ).reshape(batch_size, num_heads, kv_len, head_size)
    head_mask = np.zeros((num_heads, q_len, kv_len), dtype=np.float32)
    head_mask[1, :, 0] = 2.0
    nonpad_kv_seqlen = np.array([3, 4], dtype=np.int64)

    rank3_output, *_ = _compute_attention(
        q,
        k,
        v,
        attn_mask=head_mask,
        nonpad_kv_seqlen=nonpad_kv_seqlen,
        is_causal=True,
        left_window_size=2,
    )
    rank4_output, *_ = _compute_attention(
        q,
        k,
        v,
        attn_mask=head_mask[np.newaxis, ...],
        nonpad_kv_seqlen=nonpad_kv_seqlen,
        is_causal=True,
        left_window_size=2,
    )

    np.testing.assert_allclose(rank3_output, rank4_output)


def test_attention_asymmetric_bidirectional_window() -> None:
    q = np.zeros((1, 1, 5, 1), dtype=np.float32)
    k = np.zeros((1, 1, 5, 1), dtype=np.float32)
    v = np.arange(5, dtype=np.float32).reshape(1, 1, 5, 1)

    output, *_ = _compute_attention(
        q,
        k,
        v,
        left_window_size=1,
        right_window_size=2,
    )

    expected = np.array([1.0, 1.5, 2.5, 3.0, 3.5], dtype=np.float32)
    np.testing.assert_allclose(output.reshape(-1), expected)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("left_window_size", -2), ("right_window_size", -2)],
)
def test_attention_rejects_invalid_window_bounds(attribute, value) -> None:
    q = np.zeros((1, 1, 2, 1), dtype=np.float32)
    kwargs = {attribute: value}
    with pytest.raises(ValueError, match=rf"{attribute} must be -1 or nonnegative"):
        _compute_attention(q, q, q, **kwargs)


@pytest.mark.parametrize("left_window_size", [None, -1])
def test_attention_cache_validation_without_window(left_window_size) -> None:
    q = np.zeros((1, 2, 2, 4), dtype=np.float32)
    k = np.zeros((1, 2, 3, 4), dtype=np.float32)
    v = np.zeros((1, 2, 3, 4), dtype=np.float32)
    past_key = np.zeros((1, 2, 1, 4), dtype=np.float32)
    past_value = np.zeros((1, 2, 1, 4), dtype=np.float32)

    with pytest.raises(
        ValueError, match="past_key and past_value must be provided together"
    ):
        _compute_attention(
            q, k, v, past_key=past_key, left_window_size=left_window_size
        )

    with pytest.raises(
        ValueError,
        match="nonpad_kv_seqlen cannot be combined with past cache tensors",
    ):
        _compute_attention(
            q,
            k,
            v,
            past_key=past_key,
            past_value=past_value,
            nonpad_kv_seqlen=np.array([3], dtype=np.int64),
            left_window_size=left_window_size,
        )


def test_attention_cache_validation_is_preserved_for_older_opsets() -> None:
    q = np.zeros((1, 2, 2, 4), dtype=np.float32)
    attention_impl = load_op("", "Attention", 24)
    assert attention_impl._validate_attention25 is False
    with pytest.raises(
        ValueError, match="past_key and past_value must be provided together"
    ):
        _compute_attention(
            q,
            q,
            q,
            past_key=q,
            _validate_attention25=attention_impl._validate_attention25,
        )


def _evaluate_sink(feeds, **attrs):
    input_names = [
        "Q",
        "K",
        "V",
        "attn_mask",
        "past_key",
        "past_value",
        "nonpad_kv_seqlen",
        "head_sink",
    ]
    inputs = [name if name in feeds else "" for name in input_names]
    while not inputs[-1]:
        inputs.pop()
    expected = _compute_attention(**feeds, **attrs, _validate_attention29=True)
    outputs = ["Y", "present_key", "present_value", "qk_matmul_output"]
    if "nonpad_kv_seqlen" in feeds:
        outputs[1:3] = ["", ""]
    node = onnx.helper.make_node("Attention", inputs, outputs, **attrs)
    infos = {
        name: onnx.helper.make_tensor_value_info(
            name, onnx.helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
        )
        for name, value in feeds.items()
    }
    output_infos = [
        onnx.helper.make_tensor_value_info(
            name, onnx.helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
        )
        for name, value in zip(outputs, expected, strict=True)
        if name
    ]
    opsets = [onnx.helper.make_opsetid("", 29)]
    expansion_node = onnx.NodeProto()
    expansion_node.CopyFrom(node)
    expanded, _ = function_testcase_helper(
        expansion_node,
        [infos[name].type if name else onnx.TypeProto() for name in inputs],
        "sink",
        opset_imports=opsets,
    )
    assert len(expanded) == 1
    results = []
    for nodes, imports in [([node], opsets), *expanded]:
        model = onnx.helper.make_model(
            onnx.helper.make_graph(nodes, "sink", list(infos.values()), output_infos),
            opset_imports=imports,
        )
        onnx.checker.check_model(model, full_check=True)
        result = ReferenceEvaluator(model).run(None, feeds)
        for actual, wanted in zip(
            result,
            (v for n, v in zip(outputs, expected, strict=True) if n),
            strict=True,
        ):
            np.testing.assert_allclose(actual, wanted, rtol=3e-3, atol=3e-3)
        results.append(result)
    return results[0]


@pytest.mark.parametrize("sink", [0.0, np.log(3), 1000.0, -1000.0])
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_attention_sink_analytic_and_debug_outputs(sink, mode) -> None:
    feeds = {
        "Q": np.zeros((1, 1, 1, 1), dtype=np.float32),
        "K": np.zeros((1, 1, 1, 1), dtype=np.float32),
        "V": np.full((1, 1, 1, 1), 2, dtype=np.float32),
        "head_sink": np.array([sink], dtype=np.float32),
    }
    y, key, value, scores = _evaluate_sink(feeds, qk_matmul_output_mode=mode)
    probability = np.exp(-np.logaddexp(0.0, sink))
    np.testing.assert_allclose(y, 2 * probability, atol=1e-7)
    np.testing.assert_allclose(scores, probability if mode == 3 else 0, atol=1e-7)
    np.testing.assert_array_equal(key, feeds["K"])
    np.testing.assert_array_equal(value, feeds["V"])


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("kv_len", [0, 1, 4])
@pytest.mark.parametrize("cache", ["none", "internal", "external"])
def test_attention_sink_gqa_masks_windows_and_cache(packed, kv_len, cache) -> None:
    rng = np.random.default_rng(42)
    feeds = {
        "Q": rng.normal(size=(2, 4, 2, 3)).astype(np.float32),
        "K": rng.normal(size=(2, 2, kv_len, 3)).astype(np.float32),
        "V": rng.normal(size=(2, 2, kv_len, 5)).astype(np.float32),
        "head_sink": np.array([0, 1, -2, 3], dtype=np.float32),
    }
    total = kv_len
    if cache == "internal":
        feeds["past_key"] = rng.normal(size=(2, 2, 3, 3)).astype(np.float32)
        feeds["past_value"] = rng.normal(size=(2, 2, 3, 5)).astype(np.float32)
        total += 3
    elif cache == "external":
        feeds["nonpad_kv_seqlen"] = np.array([0, kv_len], dtype=np.int64)
    mask = np.ones((4, 2, total), dtype=np.bool_)
    mask[1] = False
    feeds["attn_mask"] = mask
    attrs = {"is_causal": 1, "left_window_size": 1, "qk_matmul_output_mode": 3}
    if packed:
        for name in ("Q", "K", "V"):
            x = feeds[name]
            feeds[name] = x.transpose(0, 2, 1, 3).reshape(
                x.shape[0], x.shape[2], x.shape[1] * x.shape[3]
            )
        attrs.update(q_num_heads=4, kv_num_heads=2)
    result = _evaluate_sink(feeds, **attrs)
    probabilities = result[-1]
    assert probabilities.shape == (2, 4, 2, total)
    np.testing.assert_array_equal(probabilities[:, 1], 0)
    assert np.all(np.sum(probabilities, axis=-1) <= 1.00001)
    if total == 0:
        np.testing.assert_array_equal(result[0], 0)
    if cache == "internal":
        np.testing.assert_array_equal(result[1][:, :, :3], feeds["past_key"])
        assert result[1].shape[2] == total


@pytest.mark.parametrize(
    "precision",
    [
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.BFLOAT16,
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.DOUBLE,
    ],
)
@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_attention_sink_scaling_softcap_precision(dtype, precision, mode) -> None:
    feeds = {
        "Q": np.full((1, 2, 2, 2), 2, dtype=dtype),
        "K": np.ones((1, 1, 3, 2), dtype=dtype),
        "V": np.full((1, 1, 3, 1), 2, dtype=dtype),
        "attn_mask": np.array([0, -np.inf, np.log(2)], dtype=dtype),
        "head_sink": np.array([0, np.log(3)], dtype=dtype),
    }
    y, _, _, scores = _evaluate_sink(
        feeds,
        scale=4.0,
        softcap=2.0,
        softmax_precision=precision,
        qk_matmul_output_mode=mode,
    )
    real_logit = 2 * np.tanh(16 / 2)
    weights = np.exp(real_logit + feeds["attn_mask"].astype(np.float64))
    denominator = weights.sum() + np.exp(feeds["head_sink"].astype(np.float64))
    expected = 2 * weights.sum() / denominator
    np.testing.assert_allclose(y[0, :, 0, 0], expected, rtol=3e-3, atol=3e-3)
    if mode == 0:
        np.testing.assert_allclose(scores, 16, rtol=3e-3)
    elif mode == 1:
        np.testing.assert_allclose(scores, real_logit, rtol=3e-3)
    elif mode == 3:
        np.testing.assert_array_equal(scores[..., 1], 0)


def test_attention_sink_absence_preserves_attention25() -> None:
    rng = np.random.default_rng(7)
    q, k, v = [rng.normal(size=(1, 2, 3, 4)).astype(np.float32) for _ in range(3)]
    result = _evaluate_sink({"Q": q, "K": k, "V": v}, qk_matmul_output_mode=3)
    expected = _compute_attention(q, k, v, qk_matmul_output_mode=3)
    for actual, wanted in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_attention_sink_absence_preserves_function_nodes() -> None:
    node = onnx.helper.make_node("Attention", ["Q", "K", "V"], ["Y"])
    tensor_type = onnx.helper.make_tensor_type_proto(
        onnx.TensorProto.FLOAT, [1, 2, 3, 4]
    )
    functions = []
    for version in (25, 29):
        schema = onnx.defs.get_schema("Attention", version)
        function = onnx.FunctionProto.FromString(
            schema.get_context_dependent_function(
                node.SerializeToString(), [tensor_type.SerializeToString()] * 3
            )
        )
        assert function.node
        functions.append(function)
    assert len(functions[0].input) == 7
    assert len(functions[1].input) == 8
    assert functions[0].node == functions[1].node


@pytest.mark.parametrize("placeholder", [False, True])
def test_attention_sink_downgrade_preserves_no_sink_outputs(placeholder) -> None:
    node = onnx.helper.make_node(
        "Attention", ["Q", "K", "V"] + ([""] * 5 if placeholder else []), ["Y"]
    )
    info = [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [1, 2, 3, 4])
        for name in ("Q", "K", "V", "Y")
    ]
    model = onnx.helper.make_model(
        onnx.helper.make_graph([node], "sink", info[:3], info[3:]),
        opset_imports=[onnx.helper.make_opsetid("", 29)],
    )
    converted = onnx.version_converter.convert_version(model, 28)
    onnx.checker.check_model(converted, full_check=True)
    assert len(converted.graph.node[0].input) <= 7
    rng = np.random.default_rng(42)
    feeds = {
        value.name: rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
        for value in info[:3]
    }
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, feeds)[0],
        ReferenceEvaluator(converted).run(None, feeds)[0],
    )


@pytest.mark.parametrize(
    ("sink", "error", "message"),
    [
        (np.zeros((1, 2), dtype=np.float32), ValueError, "shape"),
        (np.zeros(1, dtype=np.float32), ValueError, "shape"),
        (np.zeros(2, dtype=np.float64), TypeError, "dtype"),
        (np.array([np.inf, 0], dtype=np.float32), ValueError, "finite"),
        (np.array([np.nan, 0], dtype=np.float32), ValueError, "finite"),
    ],
)
def test_attention_sink_invalid_reference_inputs(sink, error, message) -> None:
    q = np.zeros((1, 2, 1, 1), dtype=np.float32)
    with pytest.raises(error, match=message):
        _compute_attention(q, q, q, head_sink=sink)
