# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import checker, helper, shape_inference
from onnx.reference import ReferenceEvaluator


def _evaluate(inputs, opset=21):
    node = helper.make_node("QLinearMatMul", list(inputs), ["y"])
    graph = helper.make_graph(
        [node],
        "row_quantization",
        [
            helper.make_tensor_value_info(
                name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
            )
            for name, value in inputs.items()
        ],
        [
            helper.make_tensor_value_info(
                "y", helper.np_dtype_to_tensor_dtype(inputs["y_zero_point"].dtype), None
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    # Shape inference supplies the output shape without changing the input data.
    model = shape_inference.infer_shapes(model, strict_mode=True)
    checker.check_model(model, full_check=True)
    original = {name: value.copy() for name, value in inputs.items()}
    result = ReferenceEvaluator(model).run(None, inputs)[0]
    for name, value in inputs.items():
        np.testing.assert_array_equal(value, original[name])
    return result


def _expected(
    a, b, a_scale, a_zero_point, b_scale, b_zero_point, y_scale, y_zero_point
):
    """Compute a 2D-by-2D result with scalar Python arithmetic."""
    result = np.empty((a.shape[0], b.shape[1]), dtype=y_zero_point.dtype)
    limits = np.iinfo(result.dtype)
    for row in range(a.shape[0]):
        for column in range(b.shape[1]):
            total = sum(
                (int(a[row, inner]) - int(a_zero_point[row]))
                * (int(b[inner, column]) - int(b_zero_point[column]))
                for inner in range(a.shape[1])
            )
            scaled = (
                total * float(a_scale[row]) * float(b_scale[column]) / float(y_scale)
            )
            quantized = round(scaled) + int(y_zero_point)
            result[row, column] = min(limits.max, max(limits.min, quantized))
    return result


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 2, 2), (1, 3, 4), (3, 1, 2)])
@pytest.mark.parametrize("a_mode", ["per_row", "scalar", "size_one"])
@pytest.mark.parametrize("b_mode", ["matrix", "batched", "vector"])
@pytest.mark.parametrize("a_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("b_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize("y_dtype", [np.int8, np.uint8])
@pytest.mark.parametrize(
    ("opset", "scale_dtype"), [(10, np.float32), (21, np.float16), (21, np.float32)]
)
def test_qlinear_matmul_row_parameters(
    shape, a_mode, b_mode, a_dtype, b_dtype, y_dtype, opset, scale_dtype
):
    m, k, n = shape
    a = (np.arange(m * k).reshape(m, k) + (7 if a_dtype == np.uint8 else -9)).astype(
        a_dtype
    )
    row_scales = np.exp2(np.arange(m) - 2).astype(scale_dtype)
    row_zeros = (np.arange(m) * 2 + (3 if a_dtype == np.uint8 else -4)).astype(a_dtype)
    if a_mode == "per_row":
        a_scale, a_zero_point = row_scales, row_zeros
    else:
        row_scales[:] = 0.5
        row_zeros[:] = 3 if a_dtype == np.uint8 else -4
        parameter_shape = () if a_mode == "scalar" else (1,)
        a_scale = np.full(parameter_shape, row_scales[0], dtype=scale_dtype)
        a_zero_point = np.full(parameter_shape, row_zeros[0], dtype=a_dtype)

    y_scale = np.array(0.5, dtype=scale_dtype)
    y_zero_point = np.array(11 if y_dtype == np.uint8 else -5, dtype=y_dtype)
    batch_count = 2 if b_mode == "batched" else 1
    columns = 1 if b_mode == "vector" else n
    matrices = []
    column_scales = []
    column_zeros = []
    expected_batches = []
    for batch in range(batch_count):
        matrix = (
            np.arange(k * columns).reshape(k, columns)
            + batch * 3
            + (5 if b_dtype == np.uint8 else -7)
        ).astype(b_dtype)
        scales = np.exp2(np.arange(columns) - 2 + batch).astype(scale_dtype)
        zeros = (
            np.arange(columns) + batch + (2 if b_dtype == np.uint8 else -3)
        ).astype(b_dtype)
        matrices.append(matrix)
        column_scales.append(scales)
        column_zeros.append(zeros)
        expected_batches.append(
            _expected(
                a, matrix, row_scales, row_zeros, scales, zeros, y_scale, y_zero_point
            )
        )

    if b_mode == "vector":
        b = matrices[0][:, 0]
        b_scale = column_scales[0][0]
        b_zero_point = column_zeros[0][0]
        expected = expected_batches[0][:, 0]
    elif b_mode == "batched":
        b = np.stack(matrices)
        b_scale = np.stack(column_scales)[:, np.newaxis, :]
        b_zero_point = np.stack(column_zeros)[:, np.newaxis, :]
        expected = np.stack(expected_batches)
    else:
        b = matrices[0]
        b_scale = column_scales[0]
        b_zero_point = column_zeros[0]
        expected = expected_batches[0]
    inputs = {
        "a": a,
        "a_scale": a_scale,
        "a_zero_point": a_zero_point,
        "b": b,
        "b_scale": np.asarray(b_scale),
        "b_zero_point": np.asarray(b_zero_point),
        "y_scale": y_scale,
        "y_zero_point": y_zero_point,
    }
    actual = _evaluate(inputs, opset)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("opset", [10, 21])
@pytest.mark.parametrize("batch_shape", [(2,), (2, 3)])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
def test_qlinear_matmul_expanded_batch_parameters(opset, batch_shape, dtype):
    a = np.arange(np.prod(batch_shape) * 6).reshape(*batch_shape, 2, 3).astype(dtype)
    # Broadcast a single right-hand matrix over every left-hand batch.
    b = np.array([[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]], dtype=dtype)
    a_scale = np.broadcast_to(
        np.array([[0.25], [0.5]], np.float32), (*batch_shape, 2, 1)
    )
    a_zero_point = np.broadcast_to(np.array([[2], [4]], dtype), (*batch_shape, 2, 1))
    b_scale = np.array([0.125, 0.25, 0.5, 1.0], np.float32)
    b_zero_point = np.array([1, 2, 3, 4], dtype)
    y_scale = np.array(0.5, np.float32)
    y_zero_point = np.array(3, dtype)
    expected = np.empty((*batch_shape, 2, 4), dtype=dtype)
    for batch in np.ndindex(batch_shape):
        expected[batch] = _expected(
            a[batch],
            b,
            a_scale[batch][:, 0],
            a_zero_point[batch][:, 0],
            b_scale,
            b_zero_point,
            y_scale,
            y_zero_point,
        )
    actual = _evaluate(
        {
            "a": a,
            "a_scale": a_scale,
            "a_zero_point": a_zero_point,
            "b": b,
            "b_scale": b_scale,
            "b_zero_point": b_zero_point,
            "y_scale": y_scale,
            "y_zero_point": y_zero_point,
        },
        opset,
    )
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
