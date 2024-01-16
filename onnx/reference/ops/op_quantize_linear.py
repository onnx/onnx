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

    def common_run(  # noqa: PLR0911
        self,
        x: np.ndarray,
        y_scale: np.ndarray,
        zero_point: np.ndarray | None = None,
        axis: int = 1,
        saturate: bool = True,
    ) -> tuple[np.ndarray]:
        if len(y_scale.shape) > 1:
            raise RuntimeError("Input 2 must be a vector or a number.")
        if len(y_scale.shape) > 0 and y_scale.size == 1:
            y_scale = y_scale[0]
        if len(y_scale.shape) > 0:
            new_shape = [1 for s in x.shape]
            new_shape[axis] = len(y_scale)
            x = x / y_scale.reshape(new_shape)
        else:
            x = x / y_scale
            new_shape = x.shape  # unused
        if zero_point is not None:
            tensor_type = self.get_zero_point_type(zero_point)

            if tensor_type in _CommonQuantizeLinear.quant_integer_ranges:
                xi = np.rint(x).astype(np.int32)
                if len(y_scale.shape) > 0:
                    xi += zero_point.reshape(new_shape)
                else:
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

            raise RuntimeError(
                f"Unexpected tensor_type for input 2: tensor_type={tensor_type}, "
                f"zero_point.dtype={zero_point.dtype}."
            )

        dtype = np.uint8  # type: ignore[assignment]
        xi = np.rint(x).astype(np.int32)
        return (np.clip(xi, 0, 255).astype(dtype),)


class QuantizeLinear_10(_CommonQuantizeLinear):
    def _run(self, *args, axis=None):  # type: ignore
        # args: x, y_scale, zero_point
        return self.common_run(*args, axis=axis)  # type: ignore


class QuantizeLinear_19(_CommonQuantizeLinear):
    def _run(self, *args, axis=None, saturate=None):  # type: ignore
        # args: x, y_scale, zero_point
        return self.common_run(*args, axis=axis, saturate=saturate)  # type: ignore
