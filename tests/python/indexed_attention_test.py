# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

import onnx
from onnx import helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRunExpand
from onnx.reference.ops._op_list import load_op
from onnx.reference.ops.op_indexed_attention import _compute_indexed_attention


class IndexedAttention(OpRunExpand):
    op_domain = ""


class Attention(OpRunExpand):
    op_domain = ""


def _model(feeds, **attrs):
    inputs = [
        helper.make_tensor_value_info(
            name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
        )
        for name, value in feeds.items()
    ]
    q, v = feeds["Q"], feeds["V"]
    output = helper.make_tensor_value_info(
        "Y", inputs[0].type.tensor_type.elem_type, (*q.shape[:3], v.shape[3])
    )
    return helper.make_model(
        helper.make_graph(
            [helper.make_node("IndexedAttention", list(feeds), ["Y"], **attrs)],
            "indexed",
            inputs,
            [output],
        ),
        opset_imports=[helper.make_opsetid("", 29)],
    )


@pytest.mark.parametrize(
    "dtype", [np.float16, ml_dtypes.bfloat16, np.float32, np.float64]
)
@pytest.mark.parametrize("key_heads", [1, 2, 4])
@pytest.mark.parametrize("index_heads", [1, 4])
@pytest.mark.parametrize("scale", [None, 0.0, 0.25])
def test_indexed_attention_function_matches_dense_oracle(
    dtype, key_heads, index_heads, scale
):
    rng = np.random.default_rng(42)
    q = rng.normal(size=(2, 4, 3, 5)).astype(dtype)
    k = rng.normal(size=(2, key_heads, 7, 5)).astype(dtype)
    v = rng.normal(size=(2, key_heads, 7, 6)).astype(dtype)
    indices = np.broadcast_to(
        np.array([6, 0, 3, -1, 8], dtype=np.int64), (2, index_heads, 3, 5)
    ).copy()
    if index_heads > 1:
        indices[:, 1, :, 1] = 2
        indices[:, 3, :, 2] = 5
    # Unsorted selections ensure causality uses original key positions.
    positions = np.array([2, 4, 6], dtype=np.int64).reshape(1, 1, 3, 1)
    mask = indices <= positions
    mask[0, ..., 1, :] = False
    feeds = {"Q": q, "K": k, "V": v, "indices": indices, "candidate_mask": mask}
    attrs = {} if scale is None else {"scale": scale}
    model = _model(feeds, **attrs)
    onnx.checker.check_model(model, full_check=True)
    primary = ReferenceEvaluator(model).run(None, feeds)[0]
    expanded = ReferenceEvaluator(model, new_ops=[IndexedAttention]).run(None, feeds)[0]
    expected = np.zeros(primary.shape, dtype=np.float64)
    acc_dtype = np.float64 if dtype == np.float64 else np.float32
    split_scale = acc_dtype(np.sqrt(1 / np.sqrt(5) if scale is None else scale))
    for b in range(2):
        for h in range(4):
            kh = h // (4 // key_heads)
            ih = 0 if index_heads == 1 else h
            for t in range(3):
                selected = indices[b, ih, t]
                selected = selected[(selected >= 0) & (selected < 7) & mask[b, ih, t]]
                if selected.size == 0:
                    continue
                scores = (q[b, h, t].astype(acc_dtype) * split_scale) @ (
                    k[b, kh, selected].astype(acc_dtype) * split_scale
                ).T
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                expected[b, h, t] = weights @ v[b, kh, selected].astype(acc_dtype)
    tolerance = 0.015 if dtype == ml_dtypes.bfloat16 else 0.002
    np.testing.assert_allclose(
        primary.astype(np.float64),
        expected.astype(dtype).astype(np.float64),
        rtol=tolerance,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        expanded.astype(np.float64), primary.astype(np.float64), rtol=1e-6, atol=1e-6
    )
    assert primary.dtype == expanded.dtype == q.dtype


@pytest.mark.parametrize(("history", "count"), [(0, 0), (0, 3), (3, 0), (3, 4)])
@pytest.mark.parametrize("mask_kind", ["absent", "scalar", "head", "candidate"])
def test_indexed_attention_empty_and_fully_masked(history, count, mask_kind):
    feeds = {
        "Q": np.ones((2, 4, 2, 3), dtype=np.float32),
        "K": np.ones((2, 2, history, 3), dtype=np.float32),
        "V": np.ones((2, 2, history, 5), dtype=np.float32),
        "indices": np.full((2, 1, 2, count), -1, dtype=np.int64),
    }
    if mask_kind != "absent":
        shape = {"scalar": (), "head": (4, 1, 1), "candidate": (count,)}[mask_kind]
        feeds["candidate_mask"] = np.zeros(shape, dtype=np.bool_)
    model = _model(feeds)
    for ops in (None, [IndexedAttention], [IndexedAttention, Attention]):
        (output,) = ReferenceEvaluator(model, new_ops=ops).run(None, feeds)
        np.testing.assert_array_equal(output, np.zeros((2, 4, 2, 5), dtype=np.float32))


def test_indexed_attention_duplicate_multiset_and_extreme_logits():
    feeds = {
        "Q": np.array([[[[1000.0]]]], dtype=np.float16),
        "K": np.array([[[[1000.0], [1000.0]]]], dtype=np.float16),
        "V": np.array([[[[3.0], [12.0]]]], dtype=np.float16),
        "indices": np.array([[[[0, 0, 1]]]], dtype=np.int64),
    }
    model = _model(feeds, scale=1.0)
    for ops in (None, [IndexedAttention], [IndexedAttention, Attention]):
        (output,) = ReferenceEvaluator(model, new_ops=ops).run(None, feeds)
        np.testing.assert_array_equal(output, np.array([[[[6.0]]]], dtype=np.float16))
    assert load_op("", "IndexedAttention", 29).__name__ == "IndexedAttention_29"


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("mask_shape", [(), (5,), (3, 5), (4, 1, 1), (2, 1, 3, 5)])
def test_indexed_attention_mask_broadcast_and_index_types(index_dtype, mask_shape):
    rng = np.random.default_rng(10)
    feeds = {
        "Q": rng.normal(size=(2, 4, 3, 2)).astype(np.float32),
        "K": rng.normal(size=(2, 2, 6, 2)).astype(np.float32),
        "V": rng.normal(size=(2, 2, 6, 3)).astype(np.float32),
        "indices": np.broadcast_to(
            np.array([0, 5, 2, -1, 6], dtype=index_dtype), (2, 1, 3, 5)
        ),
        "candidate_mask": rng.integers(0, 2, size=mask_shape).astype(np.bool_),
    }
    model = _model(feeds, scale=0.5)
    primary = ReferenceEvaluator(model).run(None, feeds)[0]
    expanded = ReferenceEvaluator(model, new_ops=[IndexedAttention, Attention]).run(
        None, feeds
    )[0]
    np.testing.assert_allclose(expanded, primary, rtol=2e-6, atol=1e-6)


def test_indexed_attention_dynamic_shapes_and_reusable_indices():
    model = onnx.parser.parse_model("""
        <ir_version: 10, opset_import: ["" : 29]>
        attention (
            float[B, H, S, D] Q, float[B, G, T, D] K, float[B, G, T, E] V,
            int64[B, 1, S, C] indices
        ) => (float[B, H, S, E] Y, float[B, H, S, E] Y2) {
            Y = IndexedAttention (Q, K, V, indices)
            Y2 = IndexedAttention (Q, K, V, indices)
        }
    """)
    onnx.checker.check_model(model, full_check=True)
    evaluator = ReferenceEvaluator(model, new_ops=[IndexedAttention, Attention])
    for history, count in [(5, 3), (0, 0), (2, 4)]:
        feeds = {
            "Q": np.zeros((2, 4, 3, 2), dtype=np.float32),
            "K": np.zeros((2, 2, history, 2), dtype=np.float32),
            "V": np.ones((2, 2, history, 3), dtype=np.float32),
            "indices": np.zeros((2, 1, 3, count), dtype=np.int64),
        }
        y, y2 = evaluator.run(None, feeds)
        np.testing.assert_array_equal(y, y2)
        expected = np.full(y.shape, int(history > 0 and count > 0), dtype=np.float32)
        np.testing.assert_allclose(y, expected)


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("Q", (2, 3, 2, 3)),
        ("K", (2, 2, 4, 7)),
        ("V", (2, 2, 5, 6)),
        ("indices", (2, 3, 2, 3)),
        ("indices", (2, 1, 3, 3)),
        ("candidate_mask", (2, 4, 2, 4)),
    ],
)
def test_indexed_attention_inference_rejects_mismatched_shapes(name, shape):
    feeds = {
        "Q": np.ones((2, 4, 2, 3), dtype=np.float32),
        "K": np.ones((2, 2, 4, 3), dtype=np.float32),
        "V": np.ones((2, 2, 4, 6), dtype=np.float32),
        "indices": np.zeros((2, 1, 2, 3), dtype=np.int64),
    }
    dtype = (
        np.int64
        if name == "indices"
        else np.bool_
        if name == "candidate_mask"
        else np.float32
    )
    feeds[name] = np.zeros(shape, dtype=dtype)
    with pytest.raises(onnx.shape_inference.InferenceError):
        onnx.shape_inference.infer_shapes(_model(feeds), strict_mode=True)


def test_indexed_attention_reference_rejects_invalid_input():
    q = np.zeros((1, 1, 1, 2), dtype=np.float32)
    indices = np.zeros((1, 1, 1, 1), dtype=np.int64)
    with pytest.raises(ValueError, match="rank 4"):
        _compute_indexed_attention(q[0], q, q, indices)
    with pytest.raises(ValueError, match="int32 or int64"):
        _compute_indexed_attention(q, q, q, indices.astype(np.float32))
    with pytest.raises(ValueError, match="boolean"):
        _compute_indexed_attention(q, q, q, indices, np.ones((), dtype=np.float32))
    with pytest.raises(ValueError, match="nonnegative"):
        _compute_indexed_attention(q, q, q, indices, scale=-1.0)
