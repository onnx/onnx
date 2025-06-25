import sys
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from onnx import TensorProto
import onnx
from onnx.reference.custom_element_types import (
    bfloat16,
    float4e2m1,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)
from onnx.reference.op_run import OpRun


def typeof(data_type: TensorProto.DataType) -> np.dtype:
    # More convenient np types to work with for some specific
    # ONNX types
    custom_type_map = {
        TensorProto.BFLOAT16: bfloat16,
        TensorProto.FLOAT8E4M3FN: float8e4m3fn,
        TensorProto.FLOAT8E4M3FNUZ: float8e4m3fnuz,
        TensorProto.FLOAT8E5M2: float8e5m2,
        TensorProto.FLOAT8E5M2FNUZ: float8e5m2fnuz,
        TensorProto.FLOAT4E2M1: float4e2m1,
    }

    return custom_type_map.get(
        data_type, onnx.helper.tensor_dtype_to_np_dtype(data_type)
    )


def pack(X: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # Pack an array of 4-bit values (represented as uint8s such
    # that only the last 4 bits should be used) into
    # uint8s where a single uint8 contains two consecutive values
    # as the first or last 4-bits depending on the endianness of
    # the system.

    X = X.flatten()

    return (
        X[0::2] & 0xF | (X[1::2] & 0xF) << 4
        if sys.byteorder == "little"
        else (X[0::2] & 0xF) << 4 | X[1::2] & 0xF
    )


def unpack(X: NDArray[np.uint8 | np.int8]) -> NDArray[np.uint8]:
    # Reverse of `pack`.
    signed = X.dtype == np.int8
    X = X.view(np.uint8).flatten()
    Y = np.zeros(X.shape[:-1] + (X.shape[-1] * 2,), dtype=np.uint8)
    lo = (X & 0xF).view(np.uint8)
    hi = ((X & 0xF0) >> 4).view(np.uint8)

    if signed:
        lo[lo & 0x8 != 0] |= 0xF0
        hi[hi & 0x8 != 0] |= 0xF0

    Y[0::2] = lo if sys.byteorder == "little" else hi
    Y[1::2] = hi if sys.byteorder == "little" else lo
    return Y


class BitCast(OpRun):
    def _run(
        self, X: NDArray, to: Optional[TensorProto.DataType] = None
    ) -> tuple[NDArray]:
        if X.dtype.type in [np.str_, np.object_]:
            X = X.astype("S")
        from_size = X.itemsize
        from_shape = X.shape
        name = X.dtype.descr[0][0]

        if name in ["int4", "uint4", "float4e2m1"]:
            from_size = 0.5
            X = pack(X)

        to_data = np.frombuffer(
            X.tobytes(),
            dtype=(
                typeof(to)
                if to != TensorProto.STRING
                else "S" + str(int(X.shape[-1] * from_size))
            ),
        )
        to_size = to_data.itemsize

        if to in [TensorProto.UINT4, TensorProto.INT4, TensorProto.FLOAT4E2M1]:
            to_data = unpack(to_data).view(typeof(to))
            to_size = 0.5

        if from_size > to_size:
            # If the size of the "from" data type T1 > the size of the
            # "to" data type T2, the shape should go from [...] to
            # [..., sizeof(T1)/sizeof(T2)]
            to_data = to_data.reshape(from_shape + (int(from_size // to_size),))
        elif from_size < to_size:
            # If the size of T1 < the size of T2, reshape from
            # [..., sizeof(T2)/sizeof(T1)] to [...]
            to_data = to_data.reshape(from_shape[:-1])

        # For compatability with how onnx.numpy_helper.to_array casts
        # byte strings to regular strings
        if to == TensorProto.STRING:
            to_data = to_data.astype(np.str_)

        return (to_data,)
