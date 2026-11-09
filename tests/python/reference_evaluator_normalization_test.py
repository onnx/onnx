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


def _run(node, inputs):
    names = list(node.input)
    tensor_type = helper.np_dtype_to_tensor_dtype(inputs[0].dtype)
    graph = helper.make_graph(
        [node],
        node.op_type.lower(),
        [
            helper.make_tensor_value_info(name, tensor_type, value.shape)
            for name, value in zip(names, inputs, strict=True)
        ],
        [helper.make_tensor_value_info("Y", tensor_type, inputs[0].shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])
    checker.check_model(model)
    (result,) = ReferenceEvaluator(model).run(
        None, dict(zip(names, inputs, strict=True))
    )
    return result


def _instance_norm(x, scale, bias):
    node = helper.make_node(
        "InstanceNormalization", ["X", "S", "B"], ["Y"], epsilon=1e-5
    )
    return _run(node, [x, scale, bias])


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_instance_normalization_low_precision_large_values(dtype):
    # The variance of 0, 2, ..., 510 squares values up to 510 and overflows
    # float16 (max 65504) when computed in the input type, which turned the
    # whole output into zeros.
    x32 = np.arange(0, 512, 2, dtype=np.float32).reshape(1, 1, 256)
    scale32 = np.ones(1, dtype=np.float32)
    bias32 = np.zeros(1, dtype=np.float32)
    expected = _instance_norm(x32, scale32, bias32).astype(dtype)

    result = _instance_norm(
        x32.astype(dtype), scale32.astype(dtype), bias32.astype(dtype)
    )

    assert result.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(result.astype(np.float32)))
    assert_allclose(
        result.astype(np.float32), expected.astype(np.float32), rtol=1e-2, atol=1e-2
    )
    assert_allclose(result.astype(np.float32)[0, 0, [0, -1]], [-1.73, 1.73], atol=1e-2)


def _lrn(x):
    node = helper.make_node(
        "LRN", ["X"], ["Y"], alpha=1e-4, beta=0.75, bias=1.0, size=5
    )
    return _run(node, [x])


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_lrn_low_precision_large_values(dtype):
    # The sum of squares of 300 (90000) overflows float16 when computed in the
    # input type, which made the output zero.
    x32 = np.full((1, 5, 1, 1), 300, dtype=np.float32)
    expected = _lrn(x32).astype(dtype)

    result = _lrn(x32.astype(dtype))

    assert result.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(result.astype(np.float32)))
    assert np.all(result.astype(np.float32) > 0)
    assert_allclose(result.astype(np.float32), expected.astype(np.float32), rtol=1e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalization_float32_float64_unchanged(dtype):
    # Full precision inputs keep their own dtype for the computation.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4, 5)).astype(dtype)
    scale = rng.standard_normal(3).astype(dtype)
    bias = rng.standard_normal(3).astype(dtype)

    result = _instance_norm(x, scale, bias)
    mean = x.mean(axis=(2, 3), keepdims=True)
    var = x.var(axis=(2, 3), keepdims=True)
    expected = scale.reshape(-1, 1, 1) * (x - mean) / np.sqrt(
        var + 1e-5
    ) + bias.reshape(-1, 1, 1)
    assert result.dtype == np.dtype(dtype)
    assert_allclose(result, expected, rtol=1e-6 if dtype == np.float64 else 1e-5)

    lrn = _lrn(x)
    assert lrn.dtype == np.dtype(dtype)
