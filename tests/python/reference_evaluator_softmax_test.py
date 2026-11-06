# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_allclose

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator

_LOW_PRECISION = [np.float16, ml_dtypes.bfloat16]


def _run(node, inputs, output_shapes):
    tensor_type = helper.np_dtype_to_tensor_dtype(inputs["X"].dtype)
    graph = helper.make_graph(
        [node],
        node.op_type.lower(),
        [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in inputs.items()
        ],
        [
            helper.make_tensor_value_info(name, tensor_type, shape)
            for name, shape in output_shapes.items()
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    checker.check_model(model)
    return ReferenceEvaluator(model).run(None, inputs)


def _log_softmax(x):
    node = helper.make_node("LogSoftmax", ["X"], ["Y"], axis=-1)
    (result,) = _run(node, {"X": x}, {"Y": x.shape})
    return result


def _softmax_cross_entropy_loss(x, labels, weights=None, reduction="mean"):
    inputs = {"X": x, "labels": labels}
    if weights is not None:
        inputs["weights"] = weights
    node = helper.make_node(
        "SoftmaxCrossEntropyLoss",
        list(inputs),
        ["loss", "log_prob"],
        reduction=reduction,
    )
    loss_shape = labels.shape if reduction == "none" else ()
    return _run(node, inputs, {"loss": loss_shape, "log_prob": x.shape})


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_log_softmax_low_precision_many_classes(dtype):
    # The sum of exp over 65536 equal logits is 65536, which overflows float16
    # (max 65504) when accumulated in the input type and gave -inf; bfloat16
    # stopped accumulating at 256 and gave -5.53 instead of -11.09.
    x32 = np.zeros((1, 65536), dtype=np.float32)
    expected = _log_softmax(x32).astype(dtype)

    result = _log_softmax(x32.astype(dtype))

    assert result.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(result.astype(np.float32)))
    assert_allclose(result.astype(np.float32), expected.astype(np.float32), rtol=1e-2)
    assert_allclose(result.astype(np.float32)[0, [0, -1]], [-11.09, -11.09], rtol=1e-2)


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_softmax_cross_entropy_loss_low_precision_many_classes(dtype):
    # Same log-softmax as above: 65536 classes with zero logits gave a loss of
    # inf in float16 and 5.53 in bfloat16 instead of 11.09.
    x32 = np.zeros((2, 65536), dtype=np.float32)
    labels = np.array([0, 65535], dtype=np.int64)
    expected_loss, expected_log_prob = (
        value.astype(dtype) for value in _softmax_cross_entropy_loss(x32, labels)
    )

    loss, log_prob = _softmax_cross_entropy_loss(x32.astype(dtype), labels)

    assert loss.dtype == np.dtype(dtype)
    assert log_prob.dtype == np.dtype(dtype)
    assert np.isfinite(loss.astype(np.float32))
    assert np.all(np.isfinite(log_prob.astype(np.float32)))
    assert_allclose(
        loss.astype(np.float32), expected_loss.astype(np.float32), rtol=1e-2
    )
    assert_allclose(loss.astype(np.float32), 11.09, rtol=1e-2)
    assert_allclose(
        log_prob.astype(np.float32), expected_log_prob.astype(np.float32), rtol=1e-2
    )


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_softmax_cross_entropy_loss_low_precision_weighted(dtype, reduction):
    # The weighted path has its own reduction; it keeps the output dtype and
    # the finite loss for every reduction mode.
    x32 = np.zeros((2, 65536), dtype=np.float32)
    labels = np.array([1, 2], dtype=np.int64)
    weights32 = np.full(65536, 0.5, dtype=np.float32)
    expected_loss, _ = (
        value.astype(dtype)
        for value in _softmax_cross_entropy_loss(x32, labels, weights32, reduction)
    )

    loss, log_prob = _softmax_cross_entropy_loss(
        x32.astype(dtype), labels, weights32.astype(dtype), reduction
    )

    assert loss.dtype == np.dtype(dtype)
    assert log_prob.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(loss.astype(np.float32)))
    assert_allclose(
        loss.astype(np.float32), expected_loss.astype(np.float32), rtol=1e-2
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_softmax_float32_float64_unchanged(dtype):
    # Full precision inputs keep their own dtype for the computation.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 7)).astype(dtype)
    labels = np.array([0, 3, 6], dtype=np.int64)
    expected_log_prob = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    rtol = 1e-6 if dtype == np.float64 else 1e-5

    result = _log_softmax(x)
    assert result.dtype == np.dtype(dtype)
    assert_allclose(result, expected_log_prob, rtol=rtol)

    loss, log_prob = _softmax_cross_entropy_loss(x, labels)
    assert loss.dtype == np.dtype(dtype)
    assert log_prob.dtype == np.dtype(dtype)
    assert_allclose(log_prob, expected_log_prob, rtol=rtol)
    assert_allclose(loss, -expected_log_prob[np.arange(3), labels].mean(), rtol=rtol)
