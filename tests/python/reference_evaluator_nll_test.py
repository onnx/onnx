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
_IGNORE_INDEX = -100


def _negative_log_likelihood_loss(
    x, target, weight=None, reduction="mean", ignore_index=None
):
    inputs = {"input": x, "target": target}
    if weight is not None:
        inputs["weight"] = weight
    attributes = {"reduction": reduction}
    if ignore_index is not None:
        attributes["ignore_index"] = ignore_index
    node = helper.make_node(
        "NegativeLogLikelihoodLoss", list(inputs), ["loss"], **attributes
    )
    loss_shape = target.shape if reduction == "none" else ()
    graph = helper.make_graph(
        [node],
        "negative_log_likelihood_loss",
        [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in inputs.items()
        ],
        [
            helper.make_tensor_value_info(
                "loss", helper.np_dtype_to_tensor_dtype(x.dtype), loss_shape
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    checker.check_model(model)
    (loss,) = ReferenceEvaluator(model).run(None, inputs)
    return loss


def _many_targets(n=2, c=5, d=36000):
    # 72000 targets. The input of class c is -(c + 1) / 2, so the loss of a
    # target is (target + 1) / 2, which is exact in float16 and bfloat16.
    x32 = np.broadcast_to(
        -(np.arange(c, dtype=np.float32) + 1)[None, :, None] / 2, (n, c, d)
    ).copy()
    target = (np.arange(n)[:, None] + np.arange(d)[None, :]) % c
    return x32, target.astype(np.int64)


def _expected(target, weight=None, reduction="mean", ignore_index=None):
    # float64 loss computed directly from the definition of the operator
    valid = np.ones(target.shape, dtype=bool)
    if ignore_index is not None:
        valid = target != ignore_index
    t = np.where(valid, target, 0)
    w = np.ones(5) if weight is None else weight.astype(np.float64)
    gathered = np.where(valid, w[t], 0.0)
    loss = gathered * (t + 1) / 2
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if weight is None and ignore_index is None:
        return loss.mean()
    return loss.sum() / gathered.sum()


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_negative_log_likelihood_loss_low_precision_ignore_index(dtype):
    # With ignore_index, the mean divides the loss sum by the sum of the
    # weights. Both overflowed float16 (max 65504) with 72000 targets and gave
    # nan; bfloat16 stopped accumulating and gave 4 instead of 1.5.
    x32, target = _many_targets()
    target[:, :500] = _IGNORE_INDEX
    expected = _expected(target, ignore_index=_IGNORE_INDEX)

    loss = _negative_log_likelihood_loss(
        x32.astype(dtype), target, ignore_index=_IGNORE_INDEX
    )

    assert loss.dtype == np.dtype(dtype)
    assert np.isfinite(loss.astype(np.float32))
    assert_allclose(loss.astype(np.float32), expected, rtol=1e-2)


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_negative_log_likelihood_loss_low_precision_weighted(dtype, reduction):
    # The weighted path has its own reduction. The weighted mean was nan in
    # float16 and 2 in bfloat16 instead of 1.5. The sum weights are small, so
    # that the sum (2025) is representable in float16; bfloat16 gave 16.
    x32, target = _many_targets()
    scale = 1 / 64 if reduction == "sum" else 1
    weight32 = np.array([1.0, 2.0, 0.5, 1.0, 1.5], dtype=np.float32) * scale
    expected = _expected(target, weight32, reduction)

    loss = _negative_log_likelihood_loss(
        x32.astype(dtype), target, weight32.astype(dtype), reduction
    )

    assert loss.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(loss.astype(np.float32)))
    assert_allclose(loss.astype(np.float32), expected, rtol=1e-2)


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_negative_log_likelihood_loss_low_precision_mean(dtype):
    # Without weights the mean is np.mean, which accumulates float16 in float32
    # but not bfloat16: bfloat16 gave 0.014 instead of 1.5.
    x32, target = _many_targets()

    loss = _negative_log_likelihood_loss(x32.astype(dtype), target)

    assert loss.dtype == np.dtype(dtype)
    assert_allclose(loss.astype(np.float32), _expected(target), rtol=1e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_negative_log_likelihood_loss_float32_float64_unchanged(dtype, reduction):
    # Full precision inputs keep their own dtype for the computation.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5, 4)).astype(dtype)
    target = rng.integers(0, 5, size=(3, 4)).astype(np.int64)
    target[0, 0] = _IGNORE_INDEX
    weight = np.array([1.0, 2.0, 0.5, 1.0, 1.5], dtype=dtype)
    valid = target != _IGNORE_INDEX
    t = np.where(valid, target, 0)
    gathered = np.where(valid, weight[t], 0)
    expected = -gathered * np.take_along_axis(x, t[:, None, :], axis=1)[:, 0, :]
    if reduction == "sum":
        expected = expected.sum()
    elif reduction == "mean":
        expected = expected.sum() / gathered.sum()
    rtol = 1e-12 if dtype == np.float64 else 1e-6

    loss = _negative_log_likelihood_loss(
        x, target, weight, reduction, ignore_index=_IGNORE_INDEX
    )

    assert loss.dtype == np.dtype(dtype)
    assert_allclose(loss, expected, rtol=rtol)
