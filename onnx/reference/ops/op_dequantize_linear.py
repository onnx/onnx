# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import TensorProto
from onnx.helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun


def _reshape_input(
    value: np.ndarray,
    shape: tuple[int, ...],
    axis: int,
    block_size: int | None = None,
) -> np.ndarray:
    """Reshape/Replicate scale/zero-point to be broadcastable to shape.

    Args:
        value: the array to be reshaped/replicated
        shape: the target shape
        axis: quantization axis, applicable for per-axis and blocked quantization
        block_size: size of quantization block, applicable only for blocked quantization

    Returns:
        value array after reshape/replicate according to quantization mode.
    """
    if len(value.shape) == 0:
        return value
    if len(value.shape) > 0 and value.size == 1:
        return value[0]
    if not block_size:
        assert len(value.shape) == 1
        dims = [1] * len(shape)
        try:
            dims[axis] = value.size
            return value.reshape(tuple(dims))
        except IndexError as e:
            raise IndexError(
                f"axis is out of boundary, axis={axis}, "
                f"value.shape={value.shape}, shape={shape}."
            ) from e

    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    # repeat scale to get element-wise scale
    value = np.repeat(value, repeats=block_size, axis=axis)
    if (
        shape[axis] != value.shape[axis]
    ):  # block_size does not divide x, handle the remainder block
        value = value.take(indices=range(shape[axis]), axis=axis)
    if value.shape != shape:
        raise ValueError(
            "Invalid shapes for Blocked Quantization. Input 2 shape should identical to Input 1 shape, except for one dimension, in which blocking is performed"
        )
    assert np.broadcast_shapes(shape, value.shape) == shape
    return value


class _CommonDequantizeLinear(OpRun):
    def _run(
        self,
        x: np.ndarray,
        x_scale: np.ndarray,
        x_zero_point: np.ndarray | None = None,
        axis: int = 1,
        block_size: int = 0,
        output_dtype: int | None = None,
    ):
        x_type = np_dtype_to_tensor_dtype(x.dtype)
        fp8_type = x_type in {
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        }
        if (
            x_zero_point is not None
            and not fp8_type
            and x_type != TensorProto.FLOAT4E2M1
        ):
            zero_type = np_dtype_to_tensor_dtype(x_zero_point.dtype)
            if x_type != zero_type:
                raise ValueError(
                    f"Type mismatch {x_type} != {zero_type} in DequantizeLinear."
                )

            dx = x.astype(np.float32) - _reshape_input(
                x_zero_point, x.shape, axis, block_size
            )
        else:
            if fp8_type and x_zero_point is not None:
                u_x_zero_point = x_zero_point.astype(np.uint8)
                umi = u_x_zero_point.min()
                uma = u_x_zero_point.max()
                if umi != uma or umi != np.uint8(0):
                    raise ValueError(
                        "x_zero_point is not null but should be zero for float8 types."
                    )
            dx = x.astype(np.float32)
        y = dx * _reshape_input(x_scale, x.shape, axis, block_size)
        return (
            y.astype(
                tensor_dtype_to_np_dtype(output_dtype)
                if output_dtype
                else x_scale.dtype
            ),
        )


class DequantizeLinear_19(_CommonDequantizeLinear):
    def _run(self, x, x_scale, x_zero_point=None, axis: int = 1):
        if len(x_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, x_scale, x_zero_point, axis)


class DequantizeLinear_21(_CommonDequantizeLinear):
    def _run(self, *args, axis: int = 1, block_size: int = 0):
        # args: x, y_scale, zero_point
        return super()._run(*args, axis=axis, block_size=block_size)


class DequantizeLinear_23(_CommonDequantizeLinear):
    def _run(self, *args, axis: int = 1, block_size: int = 0, output_dtype=None):
        # args: x, y_scale, zero_point
        return super()._run(
            *args, axis=axis, block_size=block_size, output_dtype=output_dtype
        )
