# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

import onnx
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator


def _run_resize(
    data: np.ndarray,
    *,
    axes: list[int] | None = None,
    roi: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
    **attributes: Any,
) -> np.ndarray:
    inputs = ["X", "", "", ""]
    initializers = []
    for index, (name, value) in enumerate(
        (("roi", roi), ("scales", scales), ("sizes", sizes)), start=1
    ):
        if value is not None:
            inputs[index] = name
            initializers.append(onnx.numpy_helper.from_array(value, name=name))

    if axes is not None:
        attributes["axes"] = axes
    node = onnx.helper.make_node("Resize", inputs, ["Y"], **attributes)
    graph = onnx.helper.make_graph(
        [node],
        "resize",
        [
            onnx.helper.make_tensor_value_info(
                "X", onnx.helper.np_dtype_to_tensor_dtype(data.dtype), None
            )
        ],
        [onnx.helper.make_tensor_value_info("Y", TensorProto.UNDEFINED, None)],
        initializer=initializers,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 19)]
    )
    return ReferenceEvaluator(model).run(None, {"X": data})[0]


def _expand(
    values: np.ndarray, axes: list[int], shape: tuple[int, ...], fill: float
) -> np.ndarray:
    expanded = np.full(len(shape), fill, dtype=values.dtype)
    expanded[axes] = values
    return expanded


@pytest.mark.parametrize(
    ("mode", "attributes"),
    [
        ("nearest", {"nearest_mode": "round_prefer_ceil"}),
        ("linear", {}),
        ("linear", {"antialias": 1}),
        ("cubic", {"cubic_coeff_a": -0.5, "exclude_outside": 1}),
        ("cubic", {"antialias": 1}),
    ],
)
@pytest.mark.parametrize("use_sizes", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.int16])
@pytest.mark.parametrize("contiguous", [False, True])
def test_resize_partial_axes_matches_full_rank(
    mode, attributes, use_sizes, dtype, contiguous
):
    data = np.arange(2 * 3 * 4 * 5, dtype=dtype).reshape(2, 3, 4, 5)
    if not contiguous:
        data = data[..., ::-1]
    assert data.flags.c_contiguous == contiguous
    axes = [3, 1]
    kwargs: dict[str, np.ndarray]
    full_kwargs: dict[str, np.ndarray]
    if use_sizes:
        sizes = np.array([7, 2], dtype=np.int64)
        kwargs = {"sizes": sizes}
        full_kwargs = {
            "sizes": _expand(sizes, axes, data.shape, 0)
            + np.array(
                [data.shape[i] if i not in axes else 0 for i in range(data.ndim)]
            )
        }
    else:
        scales = np.array([1.4, 0.75], dtype=np.float32)
        kwargs = {"scales": scales}
        full_kwargs = {"scales": _expand(scales, axes, data.shape, 1.0)}

    actual = _run_resize(data, axes=axes, mode=mode, **attributes, **kwargs)
    expected = _run_resize(data, mode=mode, **attributes, **full_kwargs)
    assert actual.dtype == data.dtype
    assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def test_resize_partial_axes_roi_matches_full_rank():
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    axes = [2, 0]
    roi = np.array([-0.2, 0.25, 1.2, 0.8], dtype=np.float32)
    sizes = np.array([5, 3], dtype=np.int64)
    full_roi = np.array([0.25, 0.0, -0.2, 0.8, 1.0, 1.2], dtype=np.float32)
    full_sizes = np.array([3, 3, 5], dtype=np.int64)
    attributes = {
        "mode": "linear",
        "coordinate_transformation_mode": "tf_crop_and_resize",
        "extrapolation_value": 7.25,
    }

    actual = _run_resize(data, axes=axes, roi=roi, sizes=sizes, **attributes)
    expected = _run_resize(data, roi=full_roi, sizes=full_sizes, **attributes)
    assert_allclose(actual, expected)


@pytest.mark.parametrize("axes", [[-1], [-1, -3]])
def test_resize_normalizes_negative_axes(axes):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    sizes = np.full(len(axes), 5, dtype=np.int64)
    attributes = {"mode": "linear", "coordinate_transformation_mode": "asymmetric"}

    actual = _run_resize(data, axes=axes, sizes=sizes, **attributes)
    expected = _run_resize(
        data, axes=[axis % data.ndim for axis in axes], sizes=sizes, **attributes
    )
    assert_allclose(actual, expected)


@pytest.mark.parametrize("policy", ["not_larger", "not_smaller"])
def test_resize_partial_axes_keep_aspect_ratio(policy):
    data = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    axes = [2, 1]
    sizes = np.array([4, 6], dtype=np.int64)
    scale = min(4 / 5, 6 / 3) if policy == "not_larger" else max(4 / 5, 6 / 3)
    full_scales = np.array([1.0, scale, scale], dtype=np.float32)

    actual = _run_resize(
        data,
        axes=axes,
        sizes=sizes,
        mode="linear",
        keep_aspect_ratio_policy=policy,
    )
    expected = _run_resize(data, scales=full_scales, mode="linear")
    assert_allclose(actual, expected)


def test_resize_partial_axes_zero_dimensions():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    output = _run_resize(data, axes=[1], sizes=np.array([0]), mode="linear")
    assert output.shape == (2, 0, 4)

    empty = np.empty((0, 3, 4), dtype=np.float32)
    output = _run_resize(empty, axes=[2], sizes=np.array([5]), mode="linear")
    assert output.shape == (0, 3, 5)
