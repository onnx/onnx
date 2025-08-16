# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from ml_dtypes import (
    float4_e2m1fn,
    int4,
    uint4,
)

import onnx
from onnx import TensorProto
from onnx.reference.op_run import OpRun

if TYPE_CHECKING:
    from numpy.typing import NDArray


def pack(X: NDArray[np.uint8], is_odd_count: bool) -> NDArray[np.uint8]:
    # Pack an array of 4-bit values (represented as uint8s such
    # that only the last 4 bits should be used) into
    # uint8s where a single uint8 contains two consecutive values
    # as the first or last 4-bits depending on the endianness of
    # the system.

    X = X.view(np.uint8).flatten()

    if is_odd_count:
        if sys.byteorder == "little":
            X = np.append(X, 0)
        else:
            X = np.insert(X, -1, 0)

    return (
        X[0::2] & 0xF | (X[1::2] & 0xF) << 4
        if sys.byteorder == "little"
        else (X[0::2] & 0xF) << 4 | X[1::2] & 0xF
    ).astype(np.uint8)


def unpack(X: NDArray[np.uint8 | np.int8], is_odd_count: bool) -> NDArray[np.uint8]:
    # Reverse of `pack`.
    signed = X.dtype == np.int8
    X = X.view(np.uint8).flatten()
    Y = np.zeros((*X.shape[:-1], X.shape[-1] * 2), dtype=np.uint8)
    lo = (X & 0xF).view(np.uint8)
    hi = ((X & 0xF0) >> 4).view(np.uint8)

    if signed:
        lo[lo & 0x8 != 0] |= 0xF0
        hi[hi & 0x8 != 0] |= 0xF0

    Y[0::2] = lo if sys.byteorder == "little" else hi
    Y[1::2] = hi if sys.byteorder == "little" else lo

    if is_odd_count:
        Y = np.delete(Y, -1 if sys.byteorder == "little" else -2)

    return Y.astype(np.uint8)


class BitCast(OpRun):
    def _run(
        self, X: NDArray, to: TensorProto.DataType | None = None
    ) -> tuple[NDArray]:
        is_string = X.dtype.type in [np.str_, np.object_]
        if is_string:
            X = X.astype("S")
        from_size = X.itemsize
        from_shape = X.shape
        is_odd_count = X.size * X.itemsize % 2 != 0

        if X.dtype.type in [int4, uint4, float4_e2m1fn]:
            from_size = 0.5
            X = pack(X, is_odd_count)

        to_data = np.frombuffer(
            X.tobytes(),
            dtype=(
                onnx.helper.tensor_dtype_to_np_dtype(to)
                if to != TensorProto.STRING
                else "S" + str(int(X.shape[-1] * max(from_size, 1)))
            ),
        )
        to_size = to_data.itemsize

        if to in [TensorProto.UINT4, TensorProto.INT4, TensorProto.FLOAT4E2M1]:
            to_data = unpack(to_data, is_odd_count).view(
                onnx.helper.tensor_dtype_to_np_dtype(to)
            )
            to_size = 0.5

        if from_size > to_size:
            # If the size of the "from" data type T1 > the size of the
            # "to" data type T2, the shape should go from [...] to
            # [..., sizeof(T1)/sizeof(T2)]
            to_data = to_data.reshape((*from_shape, int(from_size // to_size)))
        elif from_size < to_size:
            # If the size of T1 < the size of T2, reshape from
            # [..., sizeof(T2)/sizeof(T1)] to [...]
            to_data = to_data.reshape(from_shape[:-1])
        else:
            # Sizes are the same - ensure that the original shape is preserved
            to_data = to_data.reshape(from_shape)

        # For compatability with how onnx.numpy_helper.to_array casts
        # byte strings to regular strings
        if to == TensorProto.STRING:
            to_data = to_data.astype(np.str_)

        return (to_data,)
