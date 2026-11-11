# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import AttributeProto, TensorProto
from onnx.helper import make_attribute, make_node, tensor_dtype_to_np_dtype
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops._quant_utils import reshape_multi_axis


def _evaluate(x, scale, zero_point=None, *, opset=29, **attrs):
    feeds = {"X": x, "S": scale}
    if zero_point is not None:
        feeds["Z"] = zero_point
    block_shape = attrs.pop("block_shape", None)
    node = make_node(
        "DequantizeLinear",
        list(feeds),
        ["Y"],
        **attrs,
    )
    if block_shape is not None:
        node.attribute.append(
            make_attribute("block_shape", block_shape, attr_type=AttributeProto.INTS)
        )
    return ReferenceEvaluator(
        node,
        opsets={"": opset},
    ).run(None, feeds)[0]


def _oracle(x, scale, block_shape, zero_point=None, output_dtype=None):
    # Independent element indexing: no expansion or prototype helpers.
    result = np.empty(x.shape, dtype=np.float32)
    for index in np.ndindex(x.shape):
        grid_index = tuple(i // b for i, b in zip(index, block_shape, strict=True))
        value = np.float32(x[index])
        if zero_point is not None:
            value -= np.float32(zero_point[grid_index])
        result[index] = value * scale[grid_index]
    return result.astype(
        tensor_dtype_to_np_dtype(output_dtype) if output_dtype else scale.dtype
    )


@pytest.mark.parametrize("opset", [21, 23, 24, 25, 28, 29])
@pytest.mark.parametrize("axis", [0, 1])
def test_standard_dequantize_rejects_compact_2d_grid(opset, axis):
    x = np.ones((256, 256), dtype=ml_dtypes.float8_e4m3fn)
    scale = np.array([[1, 2], [3, 4]], dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid shapes for Blocked Quantization"):
        _evaluate(x, scale, opset=opset, axis=axis, block_size=128)


def test_multi_axis_quadrants_and_standard_workaround():
    x = np.ones((256, 256), dtype=ml_dtypes.float8_e4m3fn)
    scale = np.array([[1, 2], [3, 4]], dtype=np.float32)
    got = _evaluate(x, scale, block_shape=[128, 128])
    assert_array_equal(got, _oracle(x, scale, [128, 128]))
    for row in range(2):
        for col in range(2):
            assert_array_equal(
                got[row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128],
                np.full((128, 128), scale[row, col], dtype=np.float32),
            )
    expanded_scale = np.repeat(scale, 128, axis=0)
    assert_array_equal(
        got,
        _evaluate(x, expanded_scale, axis=1, block_size=128),
    )


def test_multi_axis_expansion_trims_intermediate_arrays(monkeypatch):
    shape = (3,) * 5
    scale = np.arange(2**5, dtype=np.float32).reshape((2,) * 5)
    repeat = np.repeat
    intermediate_sizes = []

    def tracked_repeat(value, repeats, axis):
        expanded = repeat(value, repeats, axis=axis)
        intermediate_sizes.append(expanded.size)
        return expanded

    monkeypatch.setattr(np, "repeat", tracked_repeat)
    got = reshape_multi_axis(scale, shape, [2] * 5)

    # Only the dimension currently being expanded may retain partial-edge padding.
    assert max(intermediate_sizes) <= 2 * np.prod(shape)
    indices = np.ix_(*(np.arange(dim) // 2 for dim in shape))
    assert_array_equal(got, scale[indices])


@pytest.mark.parametrize(
    ("shape", "block_shape"),
    [
        ((256, 256), [128, 128]),
        ((259, 257), [128, 128]),  # Partial edges along both dimensions.
        ((3, 17, 19), [1, 8, 8]),  # Independent expert grids, partial edges.
        ((3, 16, 24), [1, 8, 8]),
        ((5, 7), [8, 8]),  # One grid element must retain its rank.
        ((3, 5), [2**63 - 1, 2**63 - 1]),
        ((), []),
        ((3, 5), [1, 1]),
        ((0, 5), [2, 3]),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        ml_dtypes.float8_e4m3fn,
        ml_dtypes.float8_e4m3fnuz,
        ml_dtypes.float8_e5m2,
        ml_dtypes.float8_e5m2fnuz,
    ],
)
@pytest.mark.parametrize(
    "output_dtype", [None, TensorProto.FLOAT16, TensorProto.BFLOAT16]
)
def test_multi_axis_fp8_numpy_oracle(shape, block_shape, dtype, output_dtype):
    rng = np.random.default_rng(8669)
    x = rng.uniform(-4, 4, size=shape).astype(dtype)
    grid_shape = tuple(
        (d + b - 1) // b for d, b in zip(shape, block_shape, strict=True)
    )
    scale = rng.uniform(0.1, 2, size=grid_shape).astype(np.float32)
    attrs = {"block_shape": block_shape}
    if output_dtype is not None:
        attrs["output_dtype"] = output_dtype
    got = _evaluate(x, scale, **attrs)
    expected = _oracle(x, scale, block_shape, output_dtype=output_dtype)
    assert got.shape == x.shape
    assert got.dtype == expected.dtype
    assert_array_equal(got, expected)


def test_multi_axis_integer_zero_point():
    x = np.arange(35, dtype=np.int8).reshape(5, 7)
    scale = np.array([[0.25, 0.5, 1], [2, 3, 4]], dtype=np.float32)
    zero_point = np.array([[1, -2, 3], [-4, 5, 6]], dtype=np.int8)
    got = _evaluate(x, scale, zero_point, block_shape=[3, 3])
    assert_array_equal(got, _oracle(x, scale, [3, 3], zero_point))
    with pytest.raises(ValueError, match="Type mismatch"):
        _evaluate(x, scale, zero_point.astype(np.uint8), block_shape=[3, 3])


def test_multi_axis_fp8_zero_point():
    x = np.ones((5, 7), dtype=ml_dtypes.float8_e4m3fn)
    scale = np.ones((2, 3), dtype=np.float32)
    zero_point = np.zeros(scale.shape, dtype=x.dtype)
    assert_array_equal(
        _evaluate(x, scale, zero_point, block_shape=[3, 3]),
        _evaluate(x, scale, block_shape=[3, 3]),
    )
    zero_point[0, 0] = 1
    with pytest.raises(ValueError, match="should be zero for float8"):
        _evaluate(x, scale, zero_point, block_shape=[3, 3])
    with pytest.raises(ValueError, match="grid shapes must match"):
        _evaluate(x, scale, np.zeros((2, 1), dtype=x.dtype), block_shape=[3, 3])


def test_multi_axis_fp8_empty_zero_point():
    x = np.empty((0, 5), dtype=ml_dtypes.float8_e4m3fn)
    scale = np.empty((0, 2), dtype=np.float32)
    zero_point = np.empty((0, 2), dtype=x.dtype)
    got = _evaluate(x, scale, zero_point, block_shape=[2, 3])
    assert got.shape == x.shape
    assert got.dtype == scale.dtype
    assert_array_equal(got, np.empty(x.shape, dtype=scale.dtype))


@pytest.mark.parametrize("block_shape", [[], [3], [3, 3, 3], [0, 3], [-1, 3]])
def test_multi_axis_invalid_block_shape(block_shape):
    with pytest.raises(ValueError, match="block_shape"):
        _evaluate(
            np.ones((5, 7), dtype=ml_dtypes.float8_e4m3fn),
            np.ones((2, 3), dtype=np.float32),
            block_shape=block_shape,
        )


@pytest.mark.parametrize("block_shape", [[1.5, 3], [True, 3]])
def test_multi_axis_block_sizes_must_be_integers(block_shape):
    with pytest.raises(ValueError, match="positive integers"):
        reshape_multi_axis(np.ones((2, 3)), (5, 7), block_shape)


@pytest.mark.parametrize("grid_shape", [(2,), (2, 1), (1, 3), (3, 3), (2, 4), ()])
def test_multi_axis_invalid_grid(grid_shape):
    with pytest.raises(ValueError, match="grid shape"):
        _evaluate(
            np.ones((5, 7), dtype=ml_dtypes.float8_e4m3fn),
            np.ones(grid_shape, dtype=np.float32),
            block_shape=[3, 3],
        )


@pytest.mark.parametrize("block_size", [-1, 1, 128])
def test_multi_axis_conflicting_block_size(block_size):
    with pytest.raises(ValueError, match="block_size must be zero"):
        _evaluate(
            np.ones((5, 7), dtype=np.int8),
            np.ones((2, 3), dtype=np.float32),
            block_shape=[3, 3],
            block_size=block_size,
        )


@pytest.mark.parametrize("axis", [0, 1, -1, 99])
def test_multi_axis_ignores_axis(axis):
    x = np.arange(35, dtype=np.int8).reshape(5, 7)
    scale = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    assert_array_equal(
        _evaluate(x, scale, block_shape=[3, 3], axis=axis, block_size=0),
        _oracle(x, scale, [3, 3]),
    )


@pytest.mark.parametrize("scale_dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize(
    "output_dtype", [None, TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16]
)
@pytest.mark.parametrize("mode", ["scalar", "per_axis", "blocked"])
def test_existing_modes_unchanged(mode, scale_dtype, output_dtype):
    x = np.arange(15, dtype=np.int8).reshape(3, 5)
    attrs = {} if output_dtype is None else {"output_dtype": output_dtype}
    if mode == "scalar":
        scale = np.array(0.3, dtype=scale_dtype)
    elif mode == "per_axis":
        scale = np.arange(1, 6, dtype=scale_dtype)
        attrs["axis"] = -1
    else:
        scale = np.arange(1, 7, dtype=scale_dtype).reshape(3, 2)
        attrs.update(axis=-1, block_size=3)
    zero_point = np.ones(scale.shape, dtype=x.dtype)
    assert_array_equal(
        _evaluate(x, scale, zero_point, **attrs),
        _evaluate(x, scale, zero_point, opset=28, **attrs),
    )


@pytest.mark.parametrize(
    "x_type",
    [
        TensorProto.UINT8,
        TensorProto.INT8,
        TensorProto.UINT16,
        TensorProto.INT16,
        TensorProto.INT32,
        TensorProto.UINT4,
        TensorProto.INT4,
        TensorProto.UINT2,
        TensorProto.INT2,
        TensorProto.FLOAT4E2M1,
        TensorProto.FLOAT6E2M3,
        TensorProto.FLOAT6E3M2,
    ],
)
def test_multi_axis_existing_input_types(x_type):
    x = np.ones((3, 5), dtype=tensor_dtype_to_np_dtype(x_type))
    scale = np.array([[0.25, 0.5], [1, 2]], dtype=np.float32)
    zero_point = np.zeros(scale.shape, dtype=x.dtype)
    got = _evaluate(x, scale, zero_point, block_shape=[2, 3])
    assert_array_equal(got, _oracle(x, scale, [2, 3], zero_point))
