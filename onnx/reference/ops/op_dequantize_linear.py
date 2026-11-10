# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import TensorProto
from onnx.helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun
from onnx.reference.ops._quant_utils import reshape_input as _reshape_input
from onnx.reference.ops._quant_utils import reshape_multi_axis as _reshape_multi_axis


class _CommonDequantizeLinear(OpRun):
    def _run(
        self,
        x: np.ndarray,
        x_scale: np.ndarray,
        x_zero_point: np.ndarray | None = None,
        axis: int = 1,
        block_size: int = 0,
        output_dtype: int | None = None,
        *,
        _block_shape: list[int] | None = None,
    ):
        def reshape(value):
            if _block_shape is not None:
                return _reshape_multi_axis(value, x.shape, _block_shape)
            return _reshape_input(value, x.shape, axis, block_size)

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

            dx = x.astype(np.float32) - reshape(x_zero_point)
        else:
            if fp8_type and x_zero_point is not None and x_zero_point.size > 0:
                u_x_zero_point = x_zero_point.astype(np.uint8)
                umi = u_x_zero_point.min()
                uma = u_x_zero_point.max()
                if umi != uma or umi != np.uint8(0):
                    raise ValueError(
                        "x_zero_point is not null but should be zero for float8 types."
                    )
            dx = x.astype(np.float32)
        y = dx * reshape(x_scale)
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


class DequantizeLinear_25(_CommonDequantizeLinear):
    def _run(self, *args, axis: int = 1, block_size: int = 0, output_dtype=None):
        # args: x, y_scale, zero_point
        return super()._run(
            *args, axis=axis, block_size=block_size, output_dtype=output_dtype
        )


class DequantizeLinear_29(_CommonDequantizeLinear):
    def _run(
        self,
        x,
        x_scale,
        x_zero_point=None,
        axis: int = 1,
        block_size: int = 0,
        output_dtype=None,
        block_shape=None,
    ):
        if block_shape is not None:
            if block_size != 0:
                raise ValueError(
                    "block_size must be zero when block_shape is specified."
                )
            if x_zero_point is not None and x_zero_point.shape != x_scale.shape:
                raise ValueError("Scale and zero-point grid shapes must match.")
        return super()._run(
            x,
            x_scale,
            x_zero_point,
            axis=axis,
            block_size=block_size,
            output_dtype=output_dtype,
            _block_shape=block_shape,
        )
