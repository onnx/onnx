# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from onnx import checker, helper
from onnx.reference import ReferenceEvaluator


def _coordinates(shape):
    return itertools.product(*(range(size) for size in shape))


def _spec_mean(data, indices, updates, op, axis):
    result = data.copy()
    axis %= data.ndim
    for destination in _coordinates(data.shape):
        contributions = []
        for update_coordinate in _coordinates(updates.shape):
            if op == "ScatterElements":
                target = list(update_coordinate)
                index = int(indices[update_coordinate])
                target[axis] = index if index >= 0 else index + data.shape[axis]
            else:
                k = indices.shape[-1]
                tuple_coordinate = update_coordinate[: indices.ndim - 1]
                prefix = [
                    int(indices[(*tuple_coordinate, dimension)])
                    for dimension in range(k)
                ]
                target = [
                    index if index >= 0 else index + data.shape[dimension]
                    for dimension, index in enumerate(prefix)
                ] + list(update_coordinate[indices.ndim - 1 :])
            if tuple(target) == destination:
                contributions.append(updates[update_coordinate])
        if not contributions:
            continue
        contributions.insert(0, data[destination])
        if np.issubdtype(data.dtype, np.integer):
            result[destination] = sum(int(value) for value in contributions) // len(
                contributions
            )
        else:
            values = [float(value) for value in contributions]
            positive_infinity = math.inf in values
            negative_infinity = -math.inf in values
            if any(math.isnan(value) for value in values) or (
                positive_infinity and negative_infinity
            ):
                result[destination] = math.nan
            elif positive_infinity or negative_infinity:
                result[destination] = math.inf if positive_infinity else -math.inf
            else:
                result[destination] = float(
                    sum(Fraction(value) for value in values) / len(values)
                )
    return result


@pytest.mark.parametrize(
    "op, data, indices, updates, expected",
    [
        (
            "ScatterElements",
            [1, 2, 3, 4],
            [0, 1, 0, 1, 2, 1],
            [1, 2, 3, 4, 5, 6],
            [5 / 3, 3.5, 4, 4],
        ),
        (
            "ScatterND",
            [[1, 2], [3, 4]],
            [[0], [0]],
            [[3, 4], [5, 6]],
            [[3, 4], [3, 4]],
        ),
    ],
)
def test_scatter_mean_spec_examples(op, data, indices, updates, expected):
    actual = _spec_mean(
        np.array(data, dtype=np.float64),
        np.array(indices, dtype=np.int64),
        np.array(updates, dtype=np.float64),
        op,
        0,
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "op, data_shape, indices_values, indices_shape, axis",
    [
        ("ScatterElements", (2, 3), [[0, 0, 0], [-2, 0, 0]], (2, 3), 0),
        ("ScatterElements", (2, 3), [[0, -3], [1, 1]], (2, 2), 1),
        ("ScatterElements", (2, 3), [[0, -3], [1, 1]], (2, 2), -1),
        ("ScatterElements", (2, 3), [], (2, 0), 1),
        ("ScatterElements", (0, 3), [], (0, 2), 1),
        ("ScatterND", (2, 3), [[0], [-2]], (2, 1), 0),
        ("ScatterND", (2, 3), [[0, 1], [-2, 1]], (2, 2), 0),
        ("ScatterND", (2, 3), [], (2, 0), 0),
        ("ScatterND", (2, 3), [], (0,), 0),
        ("ScatterND", (2, 3), [0, -2], (2,), 0),
        ("ScatterND", (2, 3), [], (0, 1), 0),
        ("ScatterND", (0, 3), [], (2, 0), 0),
    ],
)
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64])
def test_scatter_mean_from_spec(
    op, data_shape, indices_values, indices_shape, axis, index_dtype, dtype
):
    indices = np.array(indices_values, dtype=index_dtype).reshape(indices_shape)
    if op == "ScatterElements":
        updates_shape = indices.shape
    else:
        updates_shape = (*indices.shape[:-1], *data_shape[indices.shape[-1] :])
    data = np.arange(math.prod(data_shape)).reshape(data_shape).astype(dtype)
    updates = np.arange(math.prod(updates_shape)).reshape(updates_shape).astype(dtype)
    if dtype == np.int64:
        data -= 3
        updates -= 4
    floating = np.issubdtype(dtype, np.floating)
    if floating:
        data.flat[-1:] = -0.0
    scenarios = [(data, updates)]
    if floating:
        scenarios.extend(
            (data, np.full(updates_shape, special, dtype=dtype))
            for special in (np.inf, -np.inf, np.nan, np.finfo(dtype).max)
        )
        scenarios.extend(
            [
                (
                    np.full(data_shape, np.inf, dtype=dtype),
                    np.full(updates_shape, -np.inf, dtype=dtype),
                ),
                (np.full(data_shape, np.nan, dtype=dtype), updates),
            ]
        )
    else:
        scenarios.extend(
            (
                np.full(data_shape, limit, dtype=dtype),
                np.full(updates_shape, limit, dtype=dtype),
            )
            for limit in (np.iinfo(dtype).min, np.iinfo(dtype).max)
        )
    for data, updates in scenarios:
        expected = _spec_mean(data, indices, updates, op, axis)
        attributes = {"reduction": "mean"}
        if op == "ScatterElements":
            attributes["axis"] = axis
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        op, ["data", "indices", "updates"], ["output"], **attributes
                    )
                ],
                "scatter_mean_spec",
                [
                    helper.make_tensor_value_info(
                        name, helper.np_dtype_to_tensor_dtype(value.dtype), value.shape
                    )
                    for name, value in (
                        ("data", data),
                        ("indices", indices),
                        ("updates", updates),
                    )
                ],
                [
                    helper.make_tensor_value_info(
                        "output",
                        helper.np_dtype_to_tensor_dtype(data.dtype),
                        data.shape,
                    )
                ],
            ),
            opset_imports=[helper.make_opsetid("", 29)],
        )
        checker.check_model(model, full_check=True)
        actual = ReferenceEvaluator(model).run(
            None, {"data": data, "indices": indices, "updates": updates}
        )[0]
        assert actual.dtype == expected.dtype
        if floating:
            np.testing.assert_allclose(actual, expected, rtol=2e-6, equal_nan=True)
            untouched_negative_zeros = (expected == 0) & np.signbit(expected)
            np.testing.assert_array_equal(
                np.signbit(actual[untouched_negative_zeros]),
                np.signbit(expected[untouched_negative_zeros]),
            )
        else:
            np.testing.assert_array_equal(actual, expected)
