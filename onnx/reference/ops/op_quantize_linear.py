# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import ClassVar

import numpy as np

from onnx import TensorProto, subbyte
from onnx.helper import (
    float32_to_float8e4m3,
    float32_to_float8e5m2,
    np_dtype_to_tensor_dtype,
    tensor_dtype_to_np_dtype,
)
from onnx.reference.custom_element_types import (
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
    int4,
    uint4,
)
from onnx.reference.op_run import OpRun


class _CommonQuantizeLinear(OpRun):
    float32_to_float8e4m3 = np.vectorize(float32_to_float8e4m3)
    float32_to_float8e5m2 = np.vectorize(float32_to_float8e5m2)
    quant_integer_ranges: ClassVar[dict[TensorProto.DataType, tuple[int]]] = {
        TensorProto.UINT8: (0, 255),
        TensorProto.INT8: (-128, 127),
        TensorProto.UINT16: (0, 65535),
        TensorProto.INT16: (-32768, 32767),
    }

    def get_zero_point_type(self, zero_point: np.ndarray) -> int:
        zero_point_type = None
        if (
            zero_point.dtype == float8e4m3fn
            and zero_point.dtype.descr[0][0] == "e4m3fn"
        ):
            zero_point_type = TensorProto.FLOAT8E4M3FN
        elif (
            zero_point.dtype == float8e4m3fnuz
            and zero_point.dtype.descr[0][0] == "e4m3fnuz"
        ):
            zero_point_type = TensorProto.FLOAT8E4M3FNUZ
        elif zero_point.dtype == float8e5m2 and zero_point.dtype.descr[0][0] == "e5m2":
            zero_point_type = TensorProto.FLOAT8E5M2
        elif (
            zero_point.dtype == float8e5m2fnuz
            and zero_point.dtype.descr[0][0] == "e5m2fnuz"
        ):
            zero_point_type = TensorProto.FLOAT8E5M2FNUZ
        elif zero_point.dtype == uint4 and zero_point.dtype.descr[0][0] == "uint4":
            zero_point_type = TensorProto.UINT4
        elif zero_point.dtype == int4 and zero_point.dtype.descr[0][0] == "int4":
            zero_point_type = TensorProto.INT4
        else:
            zero_point_type = np_dtype_to_tensor_dtype(zero_point.dtype)
        return zero_point_type

    def _run(  # noqa: PLR0911
        self,
        x: np.ndarray,
        y_scale: np.ndarray,
        zero_point: np.ndarray | None = None,
        axis: int = 1,
        saturate: bool = True,
    ) -> tuple[np.ndarray]:
        if len(y_scale.shape) == 0 or y_scale.size == 1:  # per-tensor
            if len(y_scale.shape) > 0:
                y_scale = y_scale[0]
            x = x / y_scale
            new_shape = x.shape  # unused
        elif len(y_scale.shape) == 1 and y_scale.size > 1:  # per-axis
            new_shape = [1] * len(x.shape)
            new_shape[axis] = len(y_scale)
            x = x / y_scale.reshape(new_shape)
        else:  # per-block
            if len(x.shape) != len(y_scale.shape):
                raise ValueError(
                    "Input 2 must be a number, a vector or a tensor with the same rank as the input."
                )
            block_shape = np.array(x.shape) // np.array(y_scale.shape)
            if sum(block_shape != 1) != 1:
                raise ValueError("Blocked quantization is defined for 1-D blocks only.")

            # repeat scale to get elementwise scale
            if x.size % y_scale.size != 0:
                raise ValueError(
                    "Blocked quantization requires the scale dimensions to divide the input dimensions"
                )
            block_dim = np.where(block_shape != 1)[0][0]
            block_size = x.shape[block_dim] // y_scale.shape[block_dim]
            assert block_size == x.size // y_scale.size

            y_scale = np.repeat(y_scale, repeats=block_size, axis=block_dim)
            # compute
            x = x / y_scale

        if zero_point is not None:
            tensor_type = self.get_zero_point_type(zero_point)

            if tensor_type in _CommonQuantizeLinear.quant_integer_ranges:
                xi = np.rint(x).astype(np.int32)
                if y_scale.size == 1:
                    xi += zero_point
                elif len(y_scale.shape) == 1:
                    xi += zero_point.reshape(new_shape)
                else:
                    zero_point = np.repeat(
                        zero_point, repeats=block_size, axis=block_dim
                    )
                    xi += zero_point
                dtype = tensor_dtype_to_np_dtype(tensor_type)
                quant_range = _CommonQuantizeLinear.quant_integer_ranges[tensor_type]
                return (np.clip(xi, quant_range[0], quant_range[1]).astype(dtype),)

            if tensor_type == TensorProto.FLOAT8E4M3FN:
                f8 = _CommonQuantizeLinear.float32_to_float8e4m3(x, saturate=saturate)
                return (f8.astype(float8e4m3fn),)  # type: ignore[attr-defined]

            if tensor_type == TensorProto.FLOAT8E4M3FNUZ:
                f8 = _CommonQuantizeLinear.float32_to_float8e4m3(
                    x, uz=True, saturate=saturate
                )
                return (f8.astype(float8e4m3fnuz),)  # type: ignore[attr-defined]

            if tensor_type == TensorProto.FLOAT8E5M2:
                f8 = _CommonQuantizeLinear.float32_to_float8e5m2(x, saturate=saturate)
                return (f8.astype(float8e5m2),)  # type: ignore[attr-defined]

            if tensor_type == TensorProto.FLOAT8E5M2FNUZ:
                f8 = _CommonQuantizeLinear.float32_to_float8e5m2(
                    x, fn=True, uz=True, saturate=saturate
                )
                return (f8.astype(float8e5m2fnuz),)  # type: ignore[attr-defined]

            if tensor_type in (TensorProto.UINT4, TensorProto.INT4):
                xi = np.rint(x).astype(np.int32)
                if len(y_scale.shape) > 0:
                    xi += zero_point.reshape(new_shape)
                else:
                    xi += zero_point

                single_func = lambda x: subbyte.float32_to_4bit_unpacked(  # noqa: E731
                    x, signed=(tensor_type == TensorProto.INT4)
                )
                func = np.vectorize(single_func)
                i4 = func(xi)
                return (i4,)  # type: ignore[attr-defined]

            raise ValueError(
                f"Unexpected tensor_type for input 2: tensor_type={tensor_type}, "
                f"zero_point.dtype={zero_point.dtype}."
            )

        dtype = np.uint8  # type: ignore[assignment]
        xi = np.rint(x).astype(np.int32)
        return (np.clip(xi, 0, 255).astype(dtype),)


class QuantizeLinear_10(_CommonQuantizeLinear):
    def _run(self, x, y_scale, zero_point=None, axis=None):  # type: ignore
        if len(y_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, y_scale, zero_point, axis=axis)  # type: ignore


class QuantizeLinear_19(_CommonQuantizeLinear):
    def _run(self, x, y_scale, zero_point=None, axis=None, saturate=None):  # type: ignore
        if len(y_scale.shape) > 1:
            raise ValueError("Input 2 must be a vector or a number.")
        return super()._run(x, y_scale, zero_point, axis=axis, saturate=saturate)  # type: ignore


class QuantizeLinear_21(_CommonQuantizeLinear):
    def _run(self, *args, axis=None, saturate=None):  # type: ignore
        # args: x, y_scale, zero_point
        return super()._run(*args, axis=axis, saturate=saturate)  # type: ignore
