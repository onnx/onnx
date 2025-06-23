# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx import TensorProto
from onnx.helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_quantize_linear import reshape_input


class _CommonDequantizeLinear(OpRun):
    def _run(
        self,
        x: np.ndarray,
        x_scale: np.ndarray,
        x_zero_point: np.ndarray | None = None,
        axis: int | None = None,
        block_size: int | None = None,
        output_dtype: int | None = None,
    ):  # type: ignore
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

            dx = x.astype(np.float32) - reshape_input(
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
        y = dx * reshape_input(x_scale, x.shape, axis, block_size)
        return (
            y.astype(
                tensor_dtype_to_np_dtype(output_dtype)
                if output_dtype
                else x_scale.dtype
            ),
        )


class DequantizeLinear_19(_CommonDequantizeLinear):
    def _run(self, x, x_scale, x_zero_point=None, axis=None):
        if len(x_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, x_scale, x_zero_point, axis)


class DequantizeLinear_21(_CommonDequantizeLinear):
    def _run(self, *args, axis=None, block_size=None):  # type: ignore
        # args: x, y_scale, zero_point
        return super()._run(*args, axis=axis, block_size=block_size)  # type: ignore


class DequantizeLinear_23(_CommonDequantizeLinear):
    def _run(self, *args, axis=None, block_size=None, output_dtype=None):  # type: ignore
        # args: x, y_scale, zero_point
        return super()._run(
            *args, axis=axis, block_size=block_size, output_dtype=output_dtype
        )  # type: ignore
