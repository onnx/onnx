# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import AttributeProto, TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator

_FP8_TYPES = [
    TensorProto.FLOAT8E4M3FN,
    TensorProto.FLOAT8E4M3FNUZ,
    TensorProto.FLOAT8E5M2,
    TensorProto.FLOAT8E5M2FNUZ,
]
_INTEGER_RANGES = {
    TensorProto.INT8: (-128, 127),
    TensorProto.UINT8: (0, 255),
    TensorProto.INT16: (-32768, 32767),
    TensorProto.UINT16: (0, 65535),
    TensorProto.INT4: (-8, 7),
    TensorProto.UINT4: (0, 15),
    TensorProto.INT2: (-2, 1),
    TensorProto.UINT2: (0, 3),
}


def _evaluate(x, scale, zero_point=None, *, opset=29, **attrs):
    feeds = {"X": x, "S": scale}
    if zero_point is not None:
        feeds["Z"] = zero_point
    blocks = attrs.pop("block_shape", None)
    node = helper.make_node("QuantizeLinear", list(feeds), ["Y"], **attrs)
    if blocks is not None:
        node.attribute.append(
            helper.make_attribute("block_shape", blocks, attr_type=AttributeProto.INTS)
        )
    return ReferenceEvaluator(node, opsets={"": opset}).run(None, feeds)[0]


def _oracle(x, scale, blocks, output_dtype, zero_point=None, precision=None):
    # Compute each grid index independently, without the implementation's expansion.
    dtype = helper.tensor_dtype_to_np_dtype(output_dtype)
    division_dtype = (
        helper.tensor_dtype_to_np_dtype(precision) if precision else scale.dtype
    )
    result = np.empty(x.shape, dtype=dtype)
    for index in np.ndindex(x.shape):
        grid = tuple(i // b for i, b in zip(index, blocks, strict=True))
        value = np.asarray(x[index], dtype=division_dtype) / np.asarray(
            scale[grid], dtype=division_dtype
        )
        if output_dtype in _INTEGER_RANGES:
            low, high = _INTEGER_RANGES[output_dtype]
            rounded = round(float(value))
            if zero_point is not None:
                rounded += int(zero_point[grid])
            result[index] = min(high, max(low, rounded))
        else:
            # ml_dtypes supplies scalar FP8 rounding, independently of ONNX's cast helper.
            limit = float(ml_dtypes.finfo(dtype).max)
            result[index] = min(limit, max(-limit, float(value)))
    return result


@pytest.mark.parametrize(
    ("shape", "blocks"),
    [
        ((256, 256), [128, 128]),
        ((259, 257), [128, 128]),
        ((3, 17, 19), [1, 8, 8]),
        ((5, 7), [8, 8]),
        ((3, 5), [2**63 - 1, 2**63 - 1]),
        ((3, 5), [1, 1]),
        ((0, 5), [2, 3]),
        ((), []),
    ],
)
@pytest.mark.parametrize(
    "output_dtype", [*_FP8_TYPES, TensorProto.INT8, TensorProto.UINT8]
)
@pytest.mark.parametrize("with_zero_point", [False, True])
def test_multi_axis_numerical_oracle(shape, blocks, output_dtype, with_zero_point):
    rng = np.random.default_rng(8669)
    x = rng.uniform(-4, 4, size=shape).astype(np.float32)
    grid = tuple((d + b - 1) // b for d, b in zip(shape, blocks, strict=True))
    scale = rng.uniform(0.25, 2, size=grid).astype(np.float32)
    zero_point = None
    if with_zero_point:
        zero_point = np.full(
            grid,
            2 if output_dtype in _INTEGER_RANGES else 0,
            dtype=helper.tensor_dtype_to_np_dtype(output_dtype),
        )
    got = _evaluate(x, scale, zero_point, block_shape=blocks, output_dtype=output_dtype)
    expected = _oracle(x, scale, blocks, output_dtype, zero_point)
    assert got.shape == x.shape
    assert got.dtype == expected.dtype
    assert_array_equal(got.astype(np.float32), expected.astype(np.float32))


@pytest.mark.parametrize("output_dtype", list(_INTEGER_RANGES))
def test_multi_axis_integer_rounding_saturation_and_zero_points(output_dtype):
    low, high = _INTEGER_RANGES[output_dtype]
    dtype = helper.tensor_dtype_to_np_dtype(output_dtype)
    # Include ties with odd and even integer neighbors, and clipping on either side.
    x = np.array(
        [[-70000, -2.5, -1.5, -0.5, 0.5], [1.5, 2.5, 3.5, 70000, 0]], dtype=np.float32
    )
    scale = np.array([[1, 2, 1], [0.5, 1, 2]], dtype=np.float32)
    zero_point = np.array([[low, high, 0], [0, low, high]], dtype=dtype)
    got = _evaluate(x, scale, zero_point, block_shape=[1, 2])
    assert got.dtype == dtype
    assert_array_equal(got, _oracle(x, scale, [1, 2], output_dtype, zero_point))


@pytest.mark.parametrize("output_dtype", _FP8_TYPES)
@pytest.mark.parametrize("saturate", [0, 1])
def test_multi_axis_fp8_rounding_and_saturation(output_dtype, saturate):
    dtype = helper.tensor_dtype_to_np_dtype(output_dtype)
    x = np.array(
        [[1, 1.0625, 1.1875], [-1, -1.0625, -1.1875], [1e6, -1e6, 0]], dtype=np.float32
    )
    scale = np.ones((2, 2), dtype=np.float32)
    got = _evaluate(
        x, scale, block_shape=[2, 2], output_dtype=output_dtype, saturate=saturate
    )
    if saturate:
        limit = float(ml_dtypes.finfo(dtype).max)
        expected = np.clip(x, -limit, limit).astype(dtype)
    else:
        expected = x.astype(dtype)
    assert_array_equal(got.astype(np.float32), expected.astype(np.float32))


@pytest.mark.parametrize(
    "precision", [None, TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16]
)
@pytest.mark.parametrize("scale_dtype", [np.float32, np.float16, ml_dtypes.bfloat16])
def test_multi_axis_division_precision(precision, scale_dtype):
    x = np.array(
        [[1.4998, 2.5002, -1.4998], [1.0624, 1.1876, -1.0624]], dtype=np.float32
    )
    scale = np.array([[1, 0.7]], dtype=scale_dtype)
    attrs = {} if precision is None else {"precision": precision}
    for output_dtype in [TensorProto.INT8, TensorProto.FLOAT8E4M3FN]:
        got = _evaluate(
            x.astype(scale_dtype),
            scale,
            block_shape=[2, 2],
            output_dtype=output_dtype,
            **attrs,
        )
        expected = _oracle(
            x.astype(scale_dtype), scale, [2, 2], output_dtype, precision=precision
        )
        assert_array_equal(got.astype(np.float32), expected.astype(np.float32))


@pytest.mark.parametrize(
    "output_dtype", [*_FP8_TYPES, TensorProto.INT8, TensorProto.UINT8]
)
@pytest.mark.parametrize("with_zero_point", [False, True])
def test_quantize_dequantize_together(output_dtype, with_zero_point):
    rng = np.random.default_rng(8669)
    shape, blocks = (2, 9, 11), [1, 4, 5]
    scale = rng.uniform(0.5, 2, size=(2, 3, 3)).astype(np.float32)
    x = rng.uniform(-3, 3, size=shape).astype(np.float32)
    if output_dtype == TensorProto.UINT8:
        x = abs(x)
    dtype = helper.tensor_dtype_to_np_dtype(output_dtype)
    zero_point = np.zeros(scale.shape, dtype=dtype) if with_zero_point else None
    if with_zero_point and output_dtype in _INTEGER_RANGES:
        zero_point[:] = rng.integers(1, 5, size=scale.shape)
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, shape),
        helper.make_tensor_value_info("S", TensorProto.FLOAT, scale.shape),
    ]
    feeds = {"X": x, "S": scale}
    if with_zero_point:
        inputs.append(helper.make_tensor_value_info("Z", output_dtype, scale.shape))
        feeds["Z"] = zero_point
    node_inputs = list(feeds)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "QuantizeLinear",
                    node_inputs,
                    ["Q"],
                    block_shape=blocks,
                    output_dtype=output_dtype,
                ),
                helper.make_node(
                    "DequantizeLinear",
                    ["Q", *node_inputs[1:]],
                    ["Y"],
                    block_shape=blocks,
                ),
            ],
            "multi_axis_quantization",
            inputs,
            [
                helper.make_tensor_value_info("Q", output_dtype, shape),
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 29)],
    )
    checker.check_model(model, full_check=True)
    quantized, restored = ReferenceEvaluator(model).run(None, feeds)
    expected_q = _oracle(x, scale, blocks, output_dtype, zero_point)
    expected_y = np.empty(shape, dtype=np.float32)
    for index in np.ndindex(shape):
        grid = tuple(i // b for i, b in zip(index, blocks, strict=True))
        zero = np.float32(zero_point[grid]) if zero_point is not None else np.float32(0)
        expected_y[index] = (np.float32(expected_q[index]) - zero) * scale[grid]
        # Bound the loss by half the local quantization step.
        if output_dtype in _INTEGER_RANGES:
            bound = 0.5
        else:
            normalized = abs(float(x[index] / scale[grid]))
            mantissa_bits = 3 if output_dtype in _FP8_TYPES[:2] else 2
            quantum = 2.0 ** (np.floor(np.log2(normalized)) - mantissa_bits)
            bound = max(quantum, float(ml_dtypes.finfo(dtype).smallest_subnormal)) / 2
        assert (
            abs(float(expected_y[index]) - float(x[index]))
            <= bound * float(scale[grid]) + 1e-6
        )
    assert_array_equal(quantized.astype(np.float32), expected_q.astype(np.float32))
    assert_array_equal(restored, expected_y)
    assert np.any(restored != x)  # The composed operators are deliberately lossy.


@pytest.mark.parametrize("blocks", [[], [3], [3, 3, 3], [0, 3], [-1, 3]])
def test_invalid_block_shape(blocks):
    with pytest.raises(ValueError, match="block_shape"):
        _evaluate(
            np.ones((5, 7), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            block_shape=blocks,
        )


@pytest.mark.parametrize("grid", [(), (2,), (1, 3), (2, 4), (2, 1)])
def test_invalid_scale_grid(grid):
    with pytest.raises(ValueError, match="grid shape"):
        _evaluate(
            np.ones((5, 7), dtype=np.float32),
            np.ones(grid, dtype=np.float32),
            block_shape=[3, 3],
        )


@pytest.mark.parametrize("output_dtype", [TensorProto.INT8, TensorProto.FLOAT8E4M3FN])
def test_invalid_zero_point_grid_and_output_type(output_dtype):
    x, scale = np.ones((5, 7), dtype=np.float32), np.ones((2, 3), dtype=np.float32)
    dtype = helper.tensor_dtype_to_np_dtype(output_dtype)
    with pytest.raises(ValueError, match="grid shapes must match"):
        _evaluate(x, scale, np.zeros((2, 1), dtype=dtype), block_shape=[3, 3])
    with pytest.raises(ValueError, match="Mismatched output data-types"):
        _evaluate(
            x,
            scale,
            np.zeros((2, 3), dtype=dtype),
            block_shape=[3, 3],
            output_dtype=TensorProto.UINT8,
        )


@pytest.mark.parametrize("block_size", [-1, 1, 128])
def test_conflicting_block_size(block_size):
    with pytest.raises(ValueError, match="block_size must be zero"):
        _evaluate(
            np.ones((5, 7), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            block_shape=[3, 3],
            block_size=block_size,
        )


@pytest.mark.parametrize("axis", [0, 1, -1, 99])
def test_multi_axis_ignores_axis(axis):
    x = np.arange(35, dtype=np.float32).reshape(5, 7)
    scale = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    got = _evaluate(x, scale, block_shape=[3, 3], axis=axis)
    assert got.dtype == np.uint8
    assert_array_equal(got, _oracle(x, scale, [3, 3], TensorProto.UINT8))


@pytest.mark.parametrize("mode", ["scalar", "per_axis", "blocked"])
@pytest.mark.parametrize(
    "output_dtype",
    _FP8_TYPES
    + list(_INTEGER_RANGES)
    + [TensorProto.FLOAT4E2M1, TensorProto.FLOAT6E2M3, TensorProto.FLOAT6E3M2],
)
@pytest.mark.parametrize("precision", [None, TensorProto.FLOAT16])
def test_existing_modes_unchanged(mode, output_dtype, precision):
    x = np.linspace(-6, 6, 15, dtype=np.float32).reshape(3, 5)
    attrs = {"output_dtype": output_dtype}
    if precision is not None:
        attrs["precision"] = precision
    if mode == "scalar":
        scale = np.array(0.3, dtype=np.float32)
    elif mode == "per_axis":
        scale = np.arange(1, 6, dtype=np.float32)
        attrs["axis"] = -1
    else:
        scale = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
        attrs.update(axis=-1, block_size=3)
    zero_point = np.zeros(
        scale.shape, dtype=helper.tensor_dtype_to_np_dtype(output_dtype)
    )
    assert_array_equal(
        _evaluate(x, scale, zero_point, **attrs).astype(np.float32),
        _evaluate(x, scale, zero_point, opset=28, **attrs).astype(np.float32),
    )


@pytest.mark.parametrize(
    "output_dtype",
    [TensorProto.FLOAT4E2M1, TensorProto.FLOAT6E2M3, TensorProto.FLOAT6E3M2],
)
def test_multi_axis_low_precision_float_outputs(output_dtype):
    x = np.linspace(-4, 4, 15, dtype=np.float32).reshape(3, 5)
    scale = np.array([[0.5, 1], [2, 4]], dtype=np.float32)
    zero_point = np.zeros(
        scale.shape, dtype=helper.tensor_dtype_to_np_dtype(output_dtype)
    )
    got = _evaluate(x, scale, zero_point, block_shape=[2, 3])
    expected = _oracle(x, scale, [2, 3], output_dtype)
    assert got.dtype == expected.dtype
    assert_array_equal(got.astype(np.float32), expected.astype(np.float32))


@pytest.mark.parametrize("x_dtype", [np.float32, np.int32])
def test_multi_axis_exponent_only_scale(x_dtype):
    x = np.arange(15, dtype=x_dtype).reshape(3, 5)
    scale = np.array([[1, 2], [4, 8]], dtype=ml_dtypes.float8_e8m0fnu)
    got = _evaluate(
        x,
        scale,
        block_shape=[2, 3],
        output_dtype=TensorProto.INT8,
        precision=TensorProto.FLOAT,
    )
    expected = _oracle(x, scale, [2, 3], TensorProto.INT8, precision=TensorProto.FLOAT)
    assert_array_equal(got, expected)
