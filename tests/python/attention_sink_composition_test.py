# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import runpy
from pathlib import Path

import numpy as np
import pytest

from onnx import TensorProto, checker, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

_EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "docsgen"
    / "source"
    / "technical"
    / "attention_sink_example.py"
)
make_sink_model = runpy.run_path(str(_EXAMPLE))["make_sink_model"]


def _oracle(logits, sinks, values):
    """Scalar float64 oracle: no concatenation or array softmax."""
    probabilities = np.zeros(logits.shape, dtype=np.float64)
    output = np.zeros((*logits.shape[:-1], values.shape[-1]), dtype=np.float64)
    for b, h, q in np.ndindex(logits.shape[:-1]):
        row = [float(x) for x in logits[b, h, q]]
        sink = float(sinks[h])
        maximum = max([sink, *row])
        weights = [math.exp(x - maximum) for x in row]
        denominator = math.fsum([math.exp(sink - maximum), *weights])
        probabilities[b, h, q] = [w / denominator for w in weights]
        for d in range(values.shape[-1]):
            output[b, h, q, d] = math.fsum(
                w * float(values[b, h, j, d]) / denominator
                for j, w in enumerate(weights)
            )
    return output, probabilities


@pytest.mark.parametrize(
    ("storage_type", "dtype", "rtol", "atol"),
    [
        (TensorProto.FLOAT16, np.float16, 2e-3, 1e-5),
        (TensorProto.FLOAT, np.float32, 2e-6, 1e-7),
        (TensorProto.DOUBLE, np.float64, 1e-13, 1e-14),
    ],
)
@pytest.mark.parametrize("key_count", [0, 1, 5])
def test_documented_graph_matches_scalar_oracle(
    storage_type, dtype, rtol, atol, key_count
):
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(2, 4, 3, key_count)).astype(dtype)
    # Two KV heads shared by consecutive pairs of query heads.
    kv_values = rng.normal(size=(2, 2, key_count, 3)).astype(dtype)
    values = np.repeat(kv_values, 2, axis=1)
    sinks = np.array([-1000, -0.5, 1.25, 1000], dtype=dtype)
    if key_count:
        logits[0, :, 0] = -np.inf
        logits[1, 0, 1, 0] = 1000
        logits[1, 1, 1, 0] = -1000
    model = make_sink_model(storage_type)
    output, probabilities = ReferenceEvaluator(model).run(
        None, {"L": logits, "S": sinks, "V": values}
    )
    expected_output, expected_probabilities = _oracle(logits, sinks, values)
    np.testing.assert_allclose(output, expected_output, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        probabilities, expected_probabilities, rtol=rtol, atol=atol
    )
    assert output.dtype == dtype
    assert probabilities.dtype == (
        np.float64 if storage_type == TensorProto.DOUBLE else np.float32
    )
    assert output.shape == (2, 4, 3, 3)
    assert probabilities.shape == (2, 4, 3, key_count)
    assert np.isfinite(output).all()
    assert np.isfinite(probabilities).all()
    if key_count:
        np.testing.assert_array_equal(output[0, :, 0], 0)
        np.testing.assert_array_equal(probabilities[0, :, 0], 0)
    else:
        np.testing.assert_array_equal(output, 0)


def test_different_heads_and_zero_sink_hand_case():
    logits = np.zeros((1, 2, 1, 2), dtype=np.float32)
    sinks = np.array([0, np.log(2)], dtype=np.float32)
    values = np.array([[[[2], [4]], [[2], [4]]]], dtype=np.float32)
    output, probabilities = ReferenceEvaluator(make_sink_model()).run(
        None, {"L": logits, "S": sinks, "V": values}
    )
    np.testing.assert_allclose(output.flatten(), [2, 1.5])
    np.testing.assert_allclose(
        probabilities.reshape(2, 2), [[1 / 3, 1 / 3], [1 / 4, 1 / 4]]
    )
    assert np.all(probabilities.sum(axis=-1) < 1)


def test_constant_sink_and_zero_value_equivalence():
    model = make_sink_model(TensorProto.DOUBLE)
    sinks = np.array([0, 1], dtype=np.float64)
    model.graph.input.remove(model.graph.input[1])
    model.graph.initializer.append(numpy_helper.from_array(sinks, "S"))
    checker.check_model(model, full_check=True)
    logits = np.array([[[[1, -2]], [[-np.inf, 3]]]], dtype=np.float64)
    values = np.array([[[[2, -1], [4, 3]], [[-3, 5], [6, 2]]]], dtype=np.float64)
    output, probabilities = ReferenceEvaluator(model).run(
        None, {"L": logits, "V": values}
    )
    expected_output, expected_probabilities = _oracle(logits, sinks, values)
    np.testing.assert_allclose(output, expected_output)
    np.testing.assert_allclose(probabilities, expected_probabilities)
    full_logits = np.concatenate((logits, sinks.reshape(1, 2, 1, 1)), axis=-1)
    weights = np.exp(full_logits - full_logits.max(axis=-1, keepdims=True))
    full_probabilities = weights / weights.sum(axis=-1, keepdims=True)
    full_values = np.concatenate((values, np.zeros((1, 2, 1, 2))), axis=-2)
    np.testing.assert_allclose(output, full_probabilities @ full_values)


@pytest.mark.parametrize("block_size", [1, 2, 5])
@pytest.mark.parametrize("key_count", [0, 5])
def test_documented_online_fusion(block_size, key_count):
    logits = np.array([1000, -np.inf, -1000, 999, -np.inf])[:key_count]
    values = np.arange(key_count * 2, dtype=np.float64).reshape(key_count, 2)
    for sink in [-1000.0, 0.0, 1000.0]:
        for row in [logits, np.full(key_count, -np.inf)]:
            maximum, denominator = sink, 1.0
            numerator = np.zeros(2, dtype=np.float64)
            for start in range(0, key_count, block_size):
                block = row[start : start + block_size]
                new_maximum = max(maximum, float(block.max()))
                rescale = math.exp(maximum - new_maximum)
                weights = np.exp(block - new_maximum)
                denominator = rescale * denominator + weights.sum()
                numerator = (
                    rescale * numerator + weights @ values[start : start + block_size]
                )
                maximum = new_maximum
            expected, _ = _oracle(
                row.reshape(1, 1, 1, key_count),
                np.array([sink]),
                values.reshape(1, 1, key_count, 2),
            )
            np.testing.assert_allclose(
                numerator / denominator, expected.flatten(), rtol=1e-14, atol=1e-14
            )


@pytest.mark.parametrize("window", [1, 3])
def test_original_position_masks_softcap_and_grouped_heads(window):
    model = make_sink_model(TensorProto.DOUBLE)
    prefix = []
    initializers = {
        "key_group_shape": np.array([1, 2, 1, 5, 2], dtype=np.int64),
        "key_expand_shape": np.array([1, 2, 2, 5, 2], dtype=np.int64),
        "key_head_shape": np.array([1, 4, 5, 2], dtype=np.int64),
        "value_group_shape": np.array([1, 2, 1, 5, 1], dtype=np.int64),
        "value_expand_shape": np.array([1, 2, 2, 5, 1], dtype=np.int64),
        "value_head_shape": np.array([1, 4, 5, 1], dtype=np.int64),
        "scale": np.array(0.5, dtype=np.float64),
        "cap": np.array(2, dtype=np.float64),
        "blocked": np.array(-np.inf, dtype=np.float64),
    }
    for source, target, stem in [("K_raw", "K_heads", "key"), ("V_raw", "V", "value")]:
        prefix.extend(
            [
                helper.make_node(
                    "Reshape", [source, f"{stem}_group_shape"], [f"{stem}_group"]
                ),
                helper.make_node(
                    "Expand",
                    [f"{stem}_group", f"{stem}_expand_shape"],
                    [f"{stem}_expanded"],
                ),
                helper.make_node(
                    "Reshape", [f"{stem}_expanded", f"{stem}_head_shape"], [target]
                ),
            ]
        )
    prefix.extend(
        [
            helper.make_node("Transpose", ["K_heads"], ["Kt"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["Q", "Kt"], ["scores"]),
            helper.make_node("Mul", ["scores", "scale"], ["scaled"]),
            helper.make_node("Div", ["scaled", "cap"], ["uncapped"]),
            helper.make_node("Tanh", ["uncapped"], ["tanh"]),
            helper.make_node("Mul", ["tanh", "cap"], ["capped"]),
            helper.make_node("Add", ["capped", "bias"], ["biased"]),
            helper.make_node("Where", ["allowed", "biased", "blocked"], ["L"]),
        ]
    )
    nodes = [*prefix, *model.graph.node]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.input[:]
    for name, dtype, shape in [
        ("Q", TensorProto.DOUBLE, [1, 4, 3, 2]),
        ("K_raw", TensorProto.DOUBLE, [1, 2, 5, 2]),
        ("V_raw", TensorProto.DOUBLE, [1, 2, 5, 1]),
        ("S", TensorProto.DOUBLE, [4]),
        ("allowed", TensorProto.BOOL, [1, 1, 3, 5]),
        ("bias", TensorProto.DOUBLE, [1, 1, 3, 5]),
    ]:
        model.graph.input.append(helper.make_tensor_value_info(name, dtype, shape))
    model.graph.initializer.extend(
        numpy_helper.from_array(value, name) for name, value in initializers.items()
    )
    checker.check_model(model, full_check=True)
    rng = np.random.default_rng(7)
    q = rng.normal(size=(1, 4, 3, 2))
    k = rng.normal(size=(1, 2, 5, 2))
    v = np.arange(10, dtype=np.float64).reshape(1, 2, 5, 1) + 1
    sinks = np.array([-3, 0, 1, 4], dtype=np.float64)
    query_positions = [8, 10, 11]
    key_positions = [4, 7, 8, 10, 11]
    valid = [True, True, False, True, True]
    allowed = np.array(
        [
            [
                qpos - window < kpos <= qpos and valid[j]
                for j, kpos in enumerate(key_positions)
            ]
            for qpos in query_positions
        ]
    )[None, None]
    bias = rng.normal(size=(1, 1, 3, 5))
    output, probabilities = ReferenceEvaluator(model).run(
        None,
        {"Q": q, "K_raw": k, "V_raw": v, "S": sinks, "allowed": allowed, "bias": bias},
    )
    # Reconstruct each score independently, including original-position eligibility.
    logits = np.full((1, 4, 3, 5), -np.inf)
    for h, i, j in np.ndindex(4, 3, 5):
        if valid[j] and 0 <= query_positions[i] - key_positions[j] < window:
            score = math.fsum(
                float(q[0, h, i, d] * k[0, h // 2, j, d]) for d in range(2)
            )
            logits[0, h, i, j] = 2 * math.tanh(score * 0.5 / 2) + bias[0, 0, i, j]
    expected_output, expected_probabilities = _oracle(
        logits, sinks, np.repeat(v, 2, axis=1)
    )
    np.testing.assert_allclose(output, expected_output, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        probabilities, expected_probabilities, rtol=1e-13, atol=1e-14
    )
    np.testing.assert_array_equal(
        probabilities[~np.broadcast_to(allowed, logits.shape)], 0
    )


def test_example_runs():
    runpy.run_path(str(_EXAMPLE), run_name="__main__")
