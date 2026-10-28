# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np
import pytest

from onnx import ModelProto, TensorProto, checker, helper, inliner, numpy_helper
from onnx import shape_inference as inference
from onnx.reference import ReferenceEvaluator


def _selected_attention_model(*, local_plus_selected: bool) -> ModelProto:
    """Fixed-shape examples described in docs/SparseAttention.md.

    B=2, Hq=4, Hkv=2, Q=3, main L=5, auxiliary L=4, D=2, Dv=1,
    selection capacity=3. Selection is shared across heads of each query.
    """
    node = helper.make_node
    inputs = []
    initializers = []
    nodes = []

    def input_info(name, dtype, shape):
        inputs.append(helper.make_tensor_value_info(name, dtype, shape))

    def constant(name, values):
        initializers.append(numpy_helper.from_array(np.array(values, np.int64), name))

    for name, shape in [
        ("Q", [2, 4, 3, 2]),
        ("K", [2, 2, 5, 2]),
        ("V", [2, 2, 5, 1]),
    ]:
        input_info(name, TensorProto.FLOAT, shape)
    for name, shape in [
        ("indices", [2, 3, 3]),
        ("counts", [2, 3]),
        ("query_positions", [2, 3]),
        ("key_positions", [2, 5]),
    ]:
        input_info(name, TensorProto.INT64, shape)
    if local_plus_selected:
        input_info("aux_K", TensorProto.FLOAT, [2, 2, 4, 2])
        input_info("aux_V", TensorProto.FLOAT, [2, 2, 4, 1])
        input_info("aux_positions", TensorProto.INT64, [2, 4])
    for name, values in [
        ("last_axis", [-1]),
        ("zero", 0),
        ("one", 1),
        ("slots", [0, 1, 2]),
        ("query_shape", [6, 4, 1, 2]),
        ("output_shape", [2, 3, 4, 1]),
    ]:
        constant(name, values)

    nodes.extend(
        [
            node("Transpose", ["Q"], ["query_bqhd"], perm=[0, 2, 1, 3]),
            node("Reshape", ["query_bqhd", "query_shape"], ["folded_Q"]),
            node("Unsqueeze", ["counts", "last_axis"], ["counts_3d"]),
            node("Less", ["slots", "counts_3d"], ["valid"]),
            # Sanitize BEFORE gathering: padding may contain arbitrary indices.
            node("Where", ["valid", "indices", "zero"], ["safe_indices"]),
            node("Unsqueeze", ["safe_indices", "last_axis"], ["selected_indices"]),
            node("Unsqueeze", ["query_positions", "last_axis"], ["query_frontier"]),
        ]
    )

    def gather_candidates(prefix, k, v, positions, indices, capacity, valid):
        for source, width in [(k, 2), (v, 1)]:
            constant(f"{prefix}_{source}_shape", [6, 2, capacity, width])
            nodes.extend(
                [
                    node("Transpose", [source], [f"{source}_blhd"], perm=[0, 2, 1, 3]),
                    node(
                        "GatherND",
                        [f"{source}_blhd", indices],
                        [f"{prefix}_{source}_bqchd"],
                        batch_dims=1,
                    ),
                    node(
                        "Transpose",
                        [f"{prefix}_{source}_bqchd"],
                        [f"{prefix}_{source}_bqhcd"],
                        perm=[0, 1, 3, 2, 4],
                    ),
                    node(
                        "Reshape",
                        [f"{prefix}_{source}_bqhcd", f"{prefix}_{source}_shape"],
                        [f"{prefix}_{source}"],
                    ),
                ]
            )
        nodes.extend(
            [
                node(
                    "GatherND",
                    [positions, indices],
                    [f"{prefix}_positions"],
                    batch_dims=1,
                ),
                node(
                    "LessOrEqual",
                    [f"{prefix}_positions", "query_frontier"],
                    [f"{prefix}_causal"],
                ),
                node("And", [f"{prefix}_causal", valid], [f"{prefix}_allowed"]),
            ]
        )

    selected_k, selected_v, selected_positions = (
        ("aux_K", "aux_V", "aux_positions")
        if local_plus_selected
        else ("K", "V", "key_positions")
    )
    gather_candidates(
        "selected",
        selected_k,
        selected_v,
        selected_positions,
        "selected_indices",
        3,
        "valid",
    )
    if local_plus_selected:
        constant(
            "local_indices", np.tile(np.arange(5).reshape(1, 1, 5, 1), (2, 3, 1, 1))
        )
        # A nonnegative distance also enforces causality; the example window is 1.
        nodes.extend(
            [
                node(
                    "GatherND",
                    ["key_positions", "local_indices"],
                    ["local_positions_raw"],
                    batch_dims=1,
                ),
                node("Sub", ["query_frontier", "local_positions_raw"], ["distance"]),
                node("GreaterOrEqual", ["distance", "zero"], ["not_future"]),
                node("LessOrEqual", ["distance", "one"], ["in_window"]),
                node("And", ["not_future", "in_window"], ["local_valid"]),
            ]
        )
        gather_candidates(
            "local", "K", "V", "key_positions", "local_indices", 5, "local_valid"
        )
        for suffix, sources in [
            ("K", ["local_K", "selected_aux_K"]),
            ("V", ["local_V", "selected_aux_V"]),
        ]:
            nodes.append(node("Concat", sources, [f"candidate_{suffix}"], axis=2))
        nodes.append(
            node("Concat", ["local_allowed", "selected_allowed"], ["allowed"], axis=2)
        )
        constant("mask_shape", [6, 1, 1, 8])
        attention_inputs = ["folded_Q", "candidate_K", "candidate_V", "mask"]
    else:
        nodes.append(node("Identity", ["selected_allowed"], ["allowed"]))
        constant("mask_shape", [6, 1, 1, 3])
        attention_inputs = ["folded_Q", "selected_K", "selected_V", "mask"]
    nodes.extend(
        [
            node("Reshape", ["allowed", "mask_shape"], ["mask"]),
            # is_causal=0: causality has already been applied in original coordinates.
            node("Attention", attention_inputs, ["folded_Y"], is_causal=0),
            node("Reshape", ["folded_Y", "output_shape"], ["output_bqhd"]),
            node("Transpose", ["output_bqhd"], ["Y"], perm=[0, 2, 1, 3]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "local_plus_selected" if local_plus_selected else "selected_only",
        inputs,
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 3, 1])],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 25)])


def _block_attention_model() -> ModelProto:
    """Expand a block-size-two support layout, including partial edge blocks."""
    node = helper.make_node
    inputs = [
        helper.make_tensor_value_info(name, dtype, shape)
        for name, dtype, shape in [
            ("Q", TensorProto.FLOAT, [2, 4, 3, 2]),
            ("K", TensorProto.FLOAT, [2, 2, 5, 2]),
            ("V", TensorProto.FLOAT, [2, 2, 5, 1]),
            ("blocks", TensorProto.BOOL, [2, 2, 2, 3]),
            ("query_positions", TensorProto.INT64, [2, 3]),
            ("key_positions", TensorProto.INT64, [2, 5]),
        ]
    ]
    initializers = [
        numpy_helper.from_array(np.array(values, np.int64), name)
        for name, values in [
            ("head_map", [0, 0, 1, 1]),
            ("query_blocks", [0, 0, 1]),
            ("key_blocks", [0, 0, 1, 1, 2]),
            ("query_axes", [1, 3]),
            ("key_axes", [1, 2]),
        ]
    ]
    nodes = [
        node("Gather", ["blocks", "head_map"], ["head_blocks"], axis=1),
        node("Gather", ["head_blocks", "query_blocks"], ["query_support"], axis=2),
        node("Gather", ["query_support", "key_blocks"], ["support"], axis=3),
        node("Unsqueeze", ["query_positions", "query_axes"], ["query_frontier"]),
        node("Unsqueeze", ["key_positions", "key_axes"], ["key_coordinates"]),
        node("LessOrEqual", ["key_coordinates", "query_frontier"], ["causal"]),
        node("And", ["support", "causal"], ["mask"]),
        node("Attention", ["Q", "K", "V", "mask"], ["Y"], is_causal=0),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "block_sparse_support",
            inputs,
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 3, 1])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 25)],
    )


def _feeds(*, local_plus_selected: bool = False) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    feeds = {
        "Q": rng.normal(size=(2, 4, 3, 2)).astype(np.float32),
        "K": rng.normal(size=(2, 2, 5, 2)).astype(np.float32),
        "V": rng.normal(size=(2, 2, 5, 1)).astype(np.float32),
        "indices": np.array(
            [
                [[2, 0, -99], [3, 1, 0], [1, 2, 3]],
                [[0, 99, -99], [2, 1, -99], [3, 0, -99]],
            ],
            np.int64,
        ),
        "counts": np.array([[2, 3, 0], [0, 2, 2]], np.int64),
        "query_positions": np.array([[1, 3, 4], [0, 2, 3]], np.int64),
        "key_positions": np.tile(np.arange(5, dtype=np.int64), (2, 1)),
    }
    if local_plus_selected:
        feeds.update(
            aux_K=rng.normal(size=(2, 2, 4, 2)).astype(np.float32),
            aux_V=rng.normal(size=(2, 2, 4, 1)).astype(np.float32),
            aux_positions=np.array([[0, 2, 4, 6], [1, 2, 3, 5]], np.int64),
        )
    return feeds


def _dense_oracle(feeds, multiplicities) -> np.ndarray:
    """Enumerate the entire source universe, without gathering or folding Q."""
    q, k, v = (feeds[name].astype(np.float64) for name in ("Q", "K", "V"))
    if "aux_K" in feeds:
        k = np.concatenate((k, feeds["aux_K"]), axis=2)
        v = np.concatenate((v, feeds["aux_V"]), axis=2)
    expected = np.zeros((2, 4, 3, 1), np.float64)
    for b in range(2):
        for h in range(4):
            kv_head = h // 2
            for i in range(3):
                counts = multiplicities[b, h, i]
                active = [j for j in range(k.shape[2]) if counts[j] > 0]
                if not active:
                    continue
                logits = [
                    np.dot(q[b, h, i], k[b, kv_head, j]) / math.sqrt(2) for j in active
                ]
                weights = [
                    math.exp(score - max(logits)) * counts[j]
                    for j, score in zip(active, logits, strict=True)
                ]
                expected[b, h, i, 0] = sum(
                    weight * v[b, kv_head, j, 0]
                    for j, weight in zip(active, weights, strict=True)
                ) / sum(weights)
    return expected


def _selected_oracle(feeds, *, local_plus_selected: bool) -> np.ndarray:
    multiplicities = np.zeros((2, 4, 3, 9 if local_plus_selected else 5), np.int64)
    for b in range(2):
        for i in range(3):
            position = feeds["query_positions"][b, i]
            if local_plus_selected:
                for j, key_position in enumerate(feeds["key_positions"][b]):
                    if 0 <= position - key_position <= 1:
                        multiplicities[b, :, i, j] = 1
            positions = feeds[
                "aux_positions" if local_plus_selected else "key_positions"
            ]
            for slot in range(feeds["counts"][b, i]):
                j = feeds["indices"][b, i, slot]
                if positions[b, j] <= position:
                    multiplicities[b, :, i, j + (5 if local_plus_selected else 0)] += 1
    return _dense_oracle(feeds, multiplicities)


def _check_and_run(model, feeds, expected, *, expanded: bool) -> np.ndarray:
    checker.check_model(model, full_check=True)
    model = inference.infer_shapes(model, strict_mode=True, data_prop=True)
    if expanded:
        model = inliner.inline_selected_functions(
            model, [("", "Attention")], inline_schema_functions=True
        )
        assert all(n.op_type != "Attention" for n in model.graph.node)
        checker.check_model(model, full_check=True)
        model = inference.infer_shapes(model, strict_mode=True, data_prop=True)
    actual = ReferenceEvaluator(model).run(None, feeds)[0]
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    return actual


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("local_plus_selected", [False, True])
def test_selected_attention(local_plus_selected: bool, expanded: bool) -> None:
    feeds = _feeds(local_plus_selected=local_plus_selected)
    _check_and_run(
        _selected_attention_model(local_plus_selected=local_plus_selected),
        feeds,
        _selected_oracle(feeds, local_plus_selected=local_plus_selected),
        expanded=expanded,
    )


@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("local_plus_selected", [False, True])
@pytest.mark.parametrize("no_local_keys", [False, True])
def test_empty_selection(
    local_plus_selected: bool, expanded: bool, no_local_keys: bool
) -> None:
    feeds = _feeds(local_plus_selected=local_plus_selected)
    feeds["counts"].fill(0)
    feeds["indices"].fill(np.iinfo(np.int64).min)
    if no_local_keys:
        feeds["key_positions"] += 10
    _check_and_run(
        _selected_attention_model(local_plus_selected=local_plus_selected),
        feeds,
        _selected_oracle(feeds, local_plus_selected=local_plus_selected),
        expanded=expanded,
    )


@pytest.mark.parametrize("expanded", [False, True])
def test_joint_normalization_and_duplicate_multiplicity(expanded: bool) -> None:
    feeds = _feeds(local_plus_selected=True)
    for name in ("Q", "K", "aux_K"):
        feeds[name].fill(0)
    feeds["V"].fill(2)
    feeds["aux_V"].fill(10)
    feeds["query_positions"].fill(2)
    feeds["aux_positions"].fill(0)
    feeds["indices"].fill(0)
    feeds["counts"].fill(1)
    model = _selected_attention_model(local_plus_selected=True)
    # Two local candidates of value 2 and one auxiliary candidate of value 10:
    # one denominator gives 14/3, not 2 + 10 from two normalized branches.
    _check_and_run(model, feeds, np.full((2, 4, 3, 1), 14 / 3), expanded=expanded)
    feeds["counts"].fill(2)
    # Repeating the auxiliary candidate contributes twice, rather than deduplicating.
    _check_and_run(model, feeds, np.full((2, 4, 3, 1), 6.0), expanded=expanded)


@pytest.mark.parametrize("expanded", [False, True])
def test_original_causal_coordinates(expanded: bool) -> None:
    feeds = _feeds()
    feeds["Q"].fill(0)
    feeds["K"].fill(0)
    feeds["V"][:] = np.arange(5, dtype=np.float32).reshape(1, 1, 5, 1) * 10
    feeds["query_positions"].fill(2)
    feeds["counts"].fill(3)
    feeds["indices"][:] = [4, 0, 2]
    # Gathered ordinal zero is a future token and ordinal two is valid.
    _check_and_run(
        _selected_attention_model(local_plus_selected=False),
        feeds,
        np.full((2, 4, 3, 1), 10.0),
        expanded=expanded,
    )


@pytest.mark.parametrize("expanded", [False, True])
def test_block_sparse_support(expanded: bool) -> None:
    feeds = _feeds()
    del feeds["indices"], feeds["counts"]
    feeds["blocks"] = np.array(
        [
            [[[1, 0, 1], [0, 1, 1]], [[0, 1, 0], [1, 0, 1]]],
            [[[0, 1, 0], [1, 0, 0]], [[1, 1, 1], [0, 0, 0]]],
        ],
        np.bool_,
    )
    multiplicities = np.zeros((2, 4, 3, 5), np.int64)
    for b in range(2):
        for h in range(4):
            for i in range(3):
                for j in range(5):
                    if (
                        feeds["blocks"][b, h // 2, i // 2, j // 2]
                        and feeds["key_positions"][b, j]
                        <= feeds["query_positions"][b, i]
                    ):
                        multiplicities[b, h, i, j] = 1
    _check_and_run(
        _block_attention_model(),
        feeds,
        _dense_oracle(feeds, multiplicities),
        expanded=expanded,
    )
