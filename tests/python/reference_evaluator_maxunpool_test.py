# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.reference import ReferenceEvaluator


def _make_model(
    x_type: int,
    x_shape: tuple[int, ...],
    *,
    output_shape: bool = False,
    pads: list[int] | None = None,
    strides: list[int] | None = None,
) -> ReferenceEvaluator:
    inputs = [
        make_tensor_value_info("X", x_type, list(x_shape)),
        make_tensor_value_info("I", TensorProto.INT64, list(x_shape)),
    ]
    node_inputs = ["X", "I"]
    if output_shape:
        inputs.append(make_tensor_value_info("S", TensorProto.INT64, [len(x_shape)]))
        node_inputs.append("S")
    attributes: dict[str, list[int]] = {
        "kernel_shape": [2] * (len(x_shape) - 2),
    }
    if pads is not None:
        attributes["pads"] = pads
    if strides is not None:
        attributes["strides"] = strides
    node = make_node("MaxUnpool", node_inputs, ["Y"], **attributes)
    graph = make_graph(
        [node],
        "maxunpool",
        inputs,
        [make_tensor_value_info("Y", x_type, None)],
    )
    return ReferenceEvaluator(make_model(graph, opset_imports=[make_opsetid("", 22)]))


@pytest.mark.parametrize(
    ("spatial_rank", "dtype", "tensor_type"),
    [
        (1, np.float16, TensorProto.FLOAT16),
        (2, np.float32, TensorProto.FLOAT),
        (3, np.float64, TensorProto.DOUBLE),
        (4, np.float32, TensorProto.FLOAT),
    ],
)
def test_max_unpool_arbitrary_spatial_rank(
    spatial_rank: int, dtype: np.dtype, tensor_type: int
) -> None:
    x_shape = (1, 1, *([1] * (spatial_rank - 1)), 2)
    x = np.array([1, 2], dtype=dtype).reshape(x_shape)
    output_shape = (1, 1, *([2] * (spatial_rank - 1)), 4)
    indices = np.array([0, np.prod(output_shape) - 1], dtype=np.int64).reshape(x_shape)

    result = _make_model(tensor_type, x_shape, strides=[2] * spatial_rank).run(
        None, {"X": x, "I": indices}
    )[0]

    expected = np.zeros(output_shape, dtype=dtype)
    expected.flat[indices.flat] = x.flat
    assert_array_equal(result, expected)


def test_max_unpool_provided_output_shape_ignores_pads() -> None:
    x = np.array([[[3, 4]]], dtype=np.float32)
    indices = np.array([[[0, 3]]], dtype=np.int64)
    output_shape = np.array([1, 1, 5], dtype=np.int64)

    result = _make_model(
        TensorProto.FLOAT,
        x.shape,
        output_shape=True,
        pads=[5, 5],
        strides=[2],
    ).run(None, {"X": x, "I": indices, "S": output_shape})[0]

    assert_array_equal(result, np.array([[[3, 0, 0, 4, 0]]], dtype=np.float32))


def test_max_unpool_inferred_shape_uses_default_strides_and_pads() -> None:
    x = np.array([[[3, 4]]], dtype=np.float32)
    indices = np.array([[[0, 2]]], dtype=np.int64)

    result = _make_model(TensorProto.FLOAT, x.shape).run(None, {"X": x, "I": indices})[0]

    assert_array_equal(result, np.array([[[3, 0, 4]]], dtype=np.float32))


def test_max_unpool_empty_input() -> None:
    x = np.empty((1, 1, 0), dtype=np.float32)
    indices = np.empty_like(x, dtype=np.int64)

    result = _make_model(TensorProto.FLOAT, x.shape).run(None, {"X": x, "I": indices})[0]

    assert_array_equal(result, np.zeros((1, 1, 1), dtype=np.float32))


def test_max_unpool_duplicate_indices_use_last_value() -> None:
    x = np.array([[[3, 4]]], dtype=np.float32)
    indices = np.array([[[1, 1]]], dtype=np.int64)

    result = _make_model(TensorProto.FLOAT, x.shape).run(None, {"X": x, "I": indices})[0]

    assert_array_equal(result, np.array([[[0, 4, 0]]], dtype=np.float32))


@pytest.mark.parametrize("indices", [np.array([[[-1]]]), np.array([[[2]]])])
def test_max_unpool_rejects_invalid_indices(indices: np.ndarray) -> None:
    x = np.array([[[1]]], dtype=np.float32)

    with pytest.raises(ValueError, match=r"Indices must be in \[0, 2\)"):
        _make_model(TensorProto.FLOAT, x.shape).run(
            None, {"X": x, "I": indices.astype(np.int64)}
        )


def test_max_unpool_rejects_indices_shape_mismatch() -> None:
    x = np.array([[[1]]], dtype=np.float32)
    indices = np.array([[[0, 1]]], dtype=np.int64)

    with pytest.raises(ValueError, match="Indices shape"):
        _make_model(TensorProto.FLOAT, x.shape).run(None, {"X": x, "I": indices})


def test_max_unpool_rejects_indices_dtype_mismatch() -> None:
    x = np.array([[[1]]], dtype=np.float32)
    indices = np.array([[[0]]], dtype=np.int32)

    with pytest.raises(TypeError):
        _make_model(TensorProto.FLOAT, x.shape).run(None, {"X": x, "I": indices})


def test_max_unpool_rejects_invalid_output_shape() -> None:
    x = np.array([[[1]]], dtype=np.float32)
    indices = np.array([[[0]]], dtype=np.int64)

    with pytest.raises(ValueError, match="same number of elements"):
        _make_model(TensorProto.FLOAT, x.shape, output_shape=True).run(
            None,
            {
                "X": x,
                "I": indices,
                "S": np.array([1, 1], dtype=np.int64),
            },
        )
