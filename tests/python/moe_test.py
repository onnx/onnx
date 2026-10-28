# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

import onnx
from onnx import helper
from onnx.inliner import inline_local_functions
from onnx.reference import ReferenceEvaluator

_EXAMPLE = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "docs" / "examples" / "moe.py")
)


def _topk(scores, k):
    return np.argsort(-scores, axis=-1, kind="stable")[..., :k]


def _router_oracle(logits, feeds, grouped):
    if grouped:
        scores = 1 / (1 + np.exp(-logits))
        choice = scores + feeds["Correction"]
        groups, width = feeds["GroupShape"]
        selected = np.zeros_like(choice, dtype=bool)
        for token in range(len(logits)):
            grouped_choice = choice[token].reshape(groups, width)
            best_two = np.sort(grouped_choice, axis=1)[:, -2:].sum(axis=1)
            eligible = _topk(best_two, int(feeds["GroupK"][0]))
            for group in eligible:
                selected[token, group * width : (group + 1) * width] = True
        choice = np.where(selected, choice, -np.inf)
    else:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        scores = np.exp(shifted)
        scores /= scores.sum(axis=1, keepdims=True)
        choice = scores
    indices = _topk(choice, int(feeds["K"][0]))
    weights = np.take_along_axis(scores, indices, axis=1)
    if feeds["Normalize"]:
        weights /= weights.sum(axis=1, keepdims=True) + feeds["Epsilon"]
    return indices, weights * feeds["Scale"]


def _swiglu(x, gate, up, down):
    activation = x @ gate
    return ((activation / (1 + np.exp(-activation))) * (x @ up)) @ down


def _oracle(feeds, grouped):
    indices, weights = _router_oracle(feeds["X"] @ feeds["Router"], feeds, grouped)
    x = feeds["X"]
    y = np.zeros_like(x)
    for token, routes in enumerate(indices):
        for route, expert in enumerate(routes):
            y[token] += weights[token, route] * _swiglu(
                x[token],
                feeds["Gate"][expert],
                feeds["Up"][expert],
                feeds["Down"][expert],
            )
    shared = _swiglu(x, feeds["SharedGate"], feeds["SharedUp"], feeds["SharedDown"])
    if not grouped:
        shared *= 1 / (1 + np.exp(-(x @ feeds["SharedRouter"])))
    return y + shared, indices, weights


def _feeds(tokens=3, experts=6, k=2, grouped=False, normalize=True):
    rng = np.random.default_rng(42)
    hidden, intermediate, shared = 4, 5, 3
    shapes = {
        "X": (tokens, hidden),
        "Router": (hidden, experts),
        "Gate": (experts, hidden, intermediate),
        "Up": (experts, hidden, intermediate),
        "Down": (experts, intermediate, hidden),
        "SharedGate": (hidden, shared),
        "SharedUp": (hidden, shared),
        "SharedDown": (shared, hidden),
    }
    feeds = {
        name: rng.normal(0, 0.3, shape).astype(np.float32)
        for name, shape in shapes.items()
    }
    feeds.update(
        K=np.array([k], dtype=np.int64),
        Normalize=np.array(normalize),
        Epsilon=np.array(1e-20 if grouped else 0, dtype=np.float32),
        Scale=np.array(1.7 if grouped else 1, dtype=np.float32),
    )
    if grouped:
        feeds.update(
            Correction=rng.normal(0, 0.2, experts).astype(np.float32),
            GroupShape=np.array([2, experts // 2], dtype=np.int64),
            GroupK=np.array([1], dtype=np.int64),
        )
    else:
        feeds["SharedRouter"] = rng.normal(0, 0.3, (hidden, 1)).astype(np.float32)
    return feeds


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("tokens", [0, 1, 4])
@pytest.mark.parametrize("k", [1, 3])
def test_moe_oracle(grouped, normalize, tokens, k):
    model = _EXAMPLE["make_model"](grouped=grouped)
    feeds = _feeds(tokens=tokens, k=k, grouped=grouped, normalize=normalize)
    actual = ReferenceEvaluator(model).run(None, feeds)
    expected = _oracle(feeds, grouped)
    for result, oracle in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, oracle, rtol=2e-5, atol=1e-6)
    assert actual[0].shape == (tokens, 4)
    assert actual[1].shape == (tokens, k)
    assert actual[1].dtype == np.int64


def _router_model(grouped):
    inputs = ["Logits", "K", "Normalize", "Epsilon", "Scale"]
    if grouped:
        inputs = ["Logits", "Correction", "GroupShape", "GroupK", *inputs[1:]]
    node = helper.make_node(
        "GroupedRouter" if grouped else "SoftmaxRouter",
        inputs,
        ["Indices", "Weights"],
        domain="local.moe",
    )
    info = {
        "Logits": (onnx.TensorProto.FLOAT, ["T", "E"]),
        "Correction": (onnx.TensorProto.FLOAT, ["E"]),
        "GroupShape": (onnx.TensorProto.INT64, [2]),
        "GroupK": (onnx.TensorProto.INT64, [1]),
        "K": (onnx.TensorProto.INT64, [1]),
        "Normalize": (onnx.TensorProto.BOOL, []),
        "Epsilon": (onnx.TensorProto.FLOAT, []),
        "Scale": (onnx.TensorProto.FLOAT, []),
    }
    graph = helper.make_graph(
        [node],
        "router",
        [helper.make_tensor_value_info(name, *info[name]) for name in inputs],
        [
            helper.make_tensor_value_info(
                "Indices", onnx.TensorProto.INT64, ["T", "k"]
            ),
            helper.make_tensor_value_info(
                "Weights", onnx.TensorProto.FLOAT, ["T", "k"]
            ),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 29),
            helper.make_opsetid("local.moe", 1),
        ],
        functions=_EXAMPLE["make_functions"](),
    )


def test_correction_only_changes_selection():
    model = _router_model(True)
    feeds = {
        "Logits": np.array([[2, 1, 0, -1]], dtype=np.float32),
        "Correction": np.zeros(4, dtype=np.float32),
        "GroupShape": np.array([2, 2], dtype=np.int64),
        "GroupK": np.array([1], dtype=np.int64),
        "K": np.array([2], dtype=np.int64),
        "Normalize": np.array(False),
        "Epsilon": np.array(1e-20, dtype=np.float32),
        "Scale": np.array(2, dtype=np.float32),
    }
    evaluator = ReferenceEvaluator(model)
    before, _ = evaluator.run(None, feeds)
    feeds["Correction"] = np.array([0, 0, 3, 3], dtype=np.float32)
    after, weights = evaluator.run(None, feeds)
    np.testing.assert_array_equal(before, [[0, 1]])
    np.testing.assert_array_equal(after, [[2, 3]])
    np.testing.assert_allclose(weights, [[1, 2 / (1 + np.exp(1))]], rtol=1e-6)


def test_negative_corrected_scores_and_epsilon():
    model = _router_model(True)
    feeds = {
        "Logits": np.array([[-50, -51, -52, -53]], dtype=np.float32),
        "Correction": np.array([-3, -3, -2, -2], dtype=np.float32),
        "GroupShape": np.array([2, 2], dtype=np.int64),
        "GroupK": np.array([1], dtype=np.int64),
        "K": np.array([2], dtype=np.int64),
        "Normalize": np.array(True),
        "Epsilon": np.array(1e-20, dtype=np.float32),
        "Scale": np.array(2, dtype=np.float32),
    }
    indices, weights = ReferenceEvaluator(model).run(None, feeds)
    expected_indices, expected_weights = _router_oracle(feeds["Logits"], feeds, True)
    np.testing.assert_array_equal(indices, [[2, 3]])
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6)
    assert weights.sum() < feeds["Scale"]


@pytest.mark.parametrize("grouped", [False, True])
def test_ties_all_groups_and_all_experts(grouped):
    model = _router_model(grouped)
    feeds = _feeds(experts=4, k=4, grouped=grouped)
    feeds["Logits"] = np.zeros((2, 4), dtype=np.float32)
    if grouped:
        feeds["Correction"][:] = 0
        feeds["GroupK"][:] = 2
    feeds = {value.name: feeds[value.name] for value in model.graph.input}
    indices, weights = ReferenceEvaluator(model).run(None, feeds)
    np.testing.assert_array_equal(indices, [[0, 1, 2, 3]] * 2)
    np.testing.assert_allclose(weights.sum(axis=1), feeds["Scale"], rtol=1e-6)


def test_group_and_expert_ties_choose_lower_index():
    model = _router_model(True)
    feeds = _feeds(experts=4, k=1, grouped=True)
    feeds["Logits"] = np.zeros((1, 4), dtype=np.float32)
    feeds["Correction"][:] = 0
    feeds = {value.name: feeds[value.name] for value in model.graph.input}
    indices, weights = ReferenceEvaluator(model).run(None, feeds)
    np.testing.assert_array_equal(indices, [[0]])
    np.testing.assert_allclose(weights, feeds["Scale"])


def test_single_expert():
    model = _EXAMPLE["make_model"]()
    feeds = _feeds(experts=1, k=1)
    actual = ReferenceEvaluator(model).run(None, feeds)
    expected = _oracle(feeds, False)
    for result, oracle in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, oracle, rtol=2e-5, atol=1e-6)


def test_duplicate_experts_are_independent_routes():
    function = _EXAMPLE["make_functions"]()[-1]
    feeds = _feeds(tokens=2, experts=1)
    inputs = ["X", "Gate", "Up", "Down", "Indices", "Weights"]
    feeds["Indices"] = np.zeros((2, 2), dtype=np.int64)
    feeds["Weights"] = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32)
    actual = ReferenceEvaluator(function).run(
        None, {name: feeds[name] for name in inputs}, attributes={}
    )[0]
    expected = _swiglu(feeds["X"], feeds["Gate"][0], feeds["Up"][0], feeds["Down"][0])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("grouped", [False, True])
def test_serialization_and_inlining(grouped):
    model = _EXAMPLE["make_model"](grouped=grouped)
    model = onnx.load_from_string(model.SerializeToString())
    model = inline_local_functions(model)
    assert not model.functions
    assert not any(node.domain == "local.moe" for node in model.graph.node)
    onnx.checker.check_model(model, full_check=True)
    feeds = _feeds(grouped=grouped)
    expected = _oracle(feeds, grouped)
    for actual, oracle in zip(
        ReferenceEvaluator(model).run(None, feeds), expected, strict=True
    ):
        np.testing.assert_allclose(actual, oracle, rtol=2e-5, atol=1e-6)
