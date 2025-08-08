# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import numpy as np

import onnx
from onnx import TensorProto
from onnx.helper import (
    np_dtype_to_tensor_dtype,
    tensor_dtype_to_np_dtype,
)
from onnx.reference.op_run import OpRun

_QUANT_TYPES = {
    TensorProto.UINT8,
    TensorProto.INT8,
    TensorProto.UINT16,
    TensorProto.INT16,
    TensorProto.UINT4,
    TensorProto.INT4,
    TensorProto.FLOAT8E4M3FN,
    TensorProto.FLOAT8E4M3FNUZ,
    TensorProto.FLOAT8E5M2,
    TensorProto.FLOAT8E5M2FNUZ,
    TensorProto.FLOAT4E2M1,
}

_QUANT_INTEGER_RANGES = {
    TensorProto.UINT8: (0, 255),
    TensorProto.INT8: (-128, 127),
    TensorProto.UINT16: (0, 65535),
    TensorProto.INT16: (-32768, 32767),
    TensorProto.UINT4: (0, 15),
    TensorProto.INT4: (-8, 7),
}


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


class _CommonQuantizeLinear(OpRun):
    def _run(
        self,
        x: np.ndarray,
        y_scale: np.ndarray,
        zero_point: np.ndarray | None = None,
        axis: int = 1,
        saturate: bool = True,
        block_size: int | None = None,
        output_dtype: TensorProto.DataType | None = None,
        precision: int | None = None,
    ) -> tuple[np.ndarray]:
        y_scale = _reshape_input(y_scale, x.shape, axis, block_size)

        # Determine output data type
        tensor_type = output_dtype
        if zero_point is not None:
            zero_point_type = np_dtype_to_tensor_dtype(zero_point.dtype)
            if output_dtype and output_dtype != zero_point_type:
                raise ValueError(
                    f"Mismatched output data-types: output_dtype={output_dtype}, zero_point type={zero_point_type}"
                )
            tensor_type = zero_point_type
        tensor_type = tensor_type or TensorProto.UINT8

        if tensor_type not in _QUANT_TYPES:
            raise ValueError(
                f"Unexpected type: output_dtype={tensor_type} is not a supported quantized type."
            )

        # Compute
        zero_point = (
            _reshape_input(zero_point, x.shape, axis, block_size)
            if zero_point is not None
            else 0
        )
        if precision:
            precision_np = tensor_dtype_to_np_dtype(precision)
            x = x.astype(precision_np) / y_scale.astype(precision_np)
        else:
            x = x / y_scale

        if tensor_type in _QUANT_INTEGER_RANGES:
            xi = np.rint(x).astype(np.int32)
            xi += zero_point
            dtype = tensor_dtype_to_np_dtype(tensor_type)
            quant_range = _QUANT_INTEGER_RANGES[tensor_type]
            return (np.clip(xi, quant_range[0], quant_range[1]).astype(dtype),)

        if tensor_type in {
            TensorProto.FLOAT8E4M3FN,
            TensorProto.FLOAT8E4M3FNUZ,
            TensorProto.FLOAT8E5M2,
            TensorProto.FLOAT8E5M2FNUZ,
        }:
            if saturate:
                return (
                    onnx.numpy_helper.saturate_cast(
                        x, dtype=tensor_dtype_to_np_dtype(tensor_type)
                    ),
                )
            else:
                return (x.astype(tensor_dtype_to_np_dtype(tensor_type)),)

        if tensor_type == TensorProto.FLOAT4E2M1:
            x += zero_point
            return (x.astype(tensor_dtype_to_np_dtype(tensor_type)),)

        raise ValueError(
            f"Unexpected type: output_dtype={tensor_type} is not a supported quantized type."
        )


class QuantizeLinear_10(_CommonQuantizeLinear):
    def _run(self, x, y_scale, zero_point=None, axis: int = 1):
        if len(y_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, y_scale, zero_point, axis=axis)


class QuantizeLinear_19(_CommonQuantizeLinear):
    def _run(self, x, y_scale, zero_point=None, axis: int = 1, saturate: bool = True):
        if len(y_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, y_scale, zero_point, axis=axis, saturate=saturate)


class QuantizeLinear_21(_CommonQuantizeLinear):
    def _run(
        self,
        *args,
        axis: int = 1,
        saturate: bool = True,
        block_size: int = 0,
        output_dtype=None,
    ):
        # args: x, y_scale, zero_point
        return super()._run(
            *args,
            axis=axis,
            saturate=saturate,
            block_size=block_size,
            output_dtype=output_dtype,
        )


class QuantizeLinear_23(_CommonQuantizeLinear):
    def _run(
        self,
        *args,
        axis: int = 1,
        saturate: bool = True,
        block_size: int = 0,
        output_dtype=None,
        precision=None,
    ):
        # args: x, y_scale, zero_point
        return super()._run(
            *args,
            axis=axis,
            saturate=saturate,
            block_size=block_size,
            output_dtype=output_dtype,
            precision=precision,
        )
