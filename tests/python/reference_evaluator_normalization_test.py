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


def _batch_norm_training(x, scale, bias, mean, var, opset=15, momentum=0.9):
    input_names = ["X", "scale", "B", "input_mean", "input_var"]
    inputs = [x, scale, bias, mean, var]
    x_type = helper.np_dtype_to_tensor_dtype(x.dtype)
    stat_type = helper.np_dtype_to_tensor_dtype(mean.dtype)
    if opset >= 14:
        output_names = ["Y", "running_mean", "running_var"]
        outputs = [
            helper.make_tensor_value_info("Y", x_type, x.shape),
            helper.make_tensor_value_info("running_mean", stat_type, mean.shape),
            helper.make_tensor_value_info("running_var", stat_type, var.shape),
        ]
        node = helper.make_node(
            "BatchNormalization",
            input_names,
            output_names,
            epsilon=1e-5,
            momentum=momentum,
            training_mode=1,
        )
    else:
        output_names = ["Y"]
        outputs = [helper.make_tensor_value_info("Y", x_type, x.shape)]
        node = helper.make_node(
            "BatchNormalization",
            input_names,
            output_names,
            epsilon=1e-5,
            momentum=momentum,
        )
    graph = helper.make_graph(
        [node],
        "batch_normalization",
        [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in zip(input_names, inputs, strict=True)
        ],
        outputs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model)
    return ReferenceEvaluator(model).run(
        None, dict(zip(input_names, inputs, strict=True))
    )


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_batch_normalization_training_low_precision_large_values(dtype):
    # The variance of 0, 2, ..., 510 squares values up to 510 and overflows
    # float16 (max 65504) when computed in the input type, which turned the
    # whole output into zeros and running_var into inf.
    x32 = np.arange(0, 512, 2, dtype=np.float32).reshape(256, 1)
    scale32 = np.ones(1, dtype=np.float32)
    bias32 = np.zeros(1, dtype=np.float32)
    mean32 = np.zeros(1, dtype=np.float32)
    var32 = np.ones(1, dtype=np.float32)
    expected_y, expected_mean, expected_var = _batch_norm_training(
        x32, scale32, bias32, mean32, var32, opset=15
    )

    y, running_mean, running_var = _batch_norm_training(
        x32.astype(dtype),
        scale32.astype(dtype),
        bias32.astype(dtype),
        mean32.astype(dtype),
        var32.astype(dtype),
        opset=15,
    )

    assert y.dtype == np.dtype(dtype)
    assert running_mean.dtype == np.dtype(dtype)
    assert running_var.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(y.astype(np.float32)))
    assert np.all(np.isfinite(running_mean.astype(np.float32)))
    assert np.all(np.isfinite(running_var.astype(np.float32)))
    assert_allclose(
        y.astype(np.float32),
        expected_y.astype(dtype).astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    assert_allclose(y.astype(np.float32)[[0, -1], 0], [-1.73, 1.73], atol=1e-2)
    assert_allclose(
        running_mean.astype(np.float32),
        expected_mean.astype(dtype).astype(np.float32),
        rtol=1e-2,
    )
    assert_allclose(
        running_var.astype(np.float32),
        expected_var.astype(dtype).astype(np.float32),
        rtol=1e-2,
    )


@pytest.mark.parametrize("dtype", _LOW_PRECISION)
def test_batch_normalization_opset9_low_precision_large_values(dtype):
    x32 = np.arange(0, 512, 2, dtype=np.float32).reshape(256, 1)
    scale32 = np.ones(1, dtype=np.float32)
    bias32 = np.zeros(1, dtype=np.float32)
    mean32 = np.zeros(1, dtype=np.float32)
    var32 = np.ones(1, dtype=np.float32)
    (expected_y,) = _batch_norm_training(
        x32, scale32, bias32, mean32, var32, opset=9
    )

    (y,) = _batch_norm_training(
        x32.astype(dtype),
        scale32.astype(dtype),
        bias32.astype(dtype),
        mean32.astype(dtype),
        var32.astype(dtype),
        opset=9,
    )

    assert y.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(y.astype(np.float32)))
    assert_allclose(
        y.astype(np.float32),
        expected_y.astype(dtype).astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batch_normalization_float32_float64_unchanged(dtype):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3, 2, 2)).astype(dtype)
    scale = rng.standard_normal(3).astype(dtype)
    bias = rng.standard_normal(3).astype(dtype)
    mean = rng.standard_normal(3).astype(dtype)
    var = np.abs(rng.standard_normal(3)).astype(dtype) + dtype(0.1)

    y, running_mean, running_var = _batch_norm_training(
        x, scale, bias, mean, var, opset=15, momentum=0.5
    )

    axis = (0, 2, 3)
    saved_mean = x.mean(axis=axis)
    saved_var = x.var(axis=axis)
    expected_mean = mean * 0.5 + saved_mean * 0.5
    expected_var = var * 0.5 + saved_var * 0.5
    expected_y = (
        scale.reshape(-1, 1, 1)
        * (x - saved_mean.reshape(-1, 1, 1))
        / np.sqrt(saved_var.reshape(-1, 1, 1) + 1e-5)
        + bias.reshape(-1, 1, 1)
    )
    rtol = 1e-12 if dtype == np.float64 else 1e-5
    y_rtol = 1e-6 if dtype == np.float64 else 1e-5
    assert y.dtype == np.dtype(dtype)
    assert running_mean.dtype == np.dtype(dtype)
    assert running_var.dtype == np.dtype(dtype)
    assert_allclose(y, expected_y, rtol=y_rtol, atol=1e-6)
    assert_allclose(running_mean, expected_mean, rtol=rtol)
    assert_allclose(running_var, expected_var, rtol=rtol)
