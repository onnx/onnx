# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from onnx import TensorProto


class TensorDtypeMap(NamedTuple):
    np_dtype: np.dtype
    storage_dtype: int
    name: str


# tensor_dtype: (numpy type, storage type, string name)
TENSOR_TYPE_MAP = {
    int(TensorProto.FLOAT): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.FLOAT), "TensorProto.FLOAT"
    ),
    int(TensorProto.UINT8): TensorDtypeMap(
        np.dtype("uint8"), int(TensorProto.INT32), "TensorProto.UINT8"
    ),
    int(TensorProto.INT8): TensorDtypeMap(
        np.dtype("int8"), int(TensorProto.INT32), "TensorProto.INT8"
    ),
    int(TensorProto.UINT16): TensorDtypeMap(
        np.dtype("uint16"), int(TensorProto.INT32), "TensorProto.UINT16"
    ),
    int(TensorProto.INT16): TensorDtypeMap(
        np.dtype("int16"), int(TensorProto.INT32), "TensorProto.INT16"
    ),
    int(TensorProto.INT32): TensorDtypeMap(
        np.dtype("int32"), int(TensorProto.INT32), "TensorProto.INT32"
    ),
    int(TensorProto.INT64): TensorDtypeMap(
        np.dtype("int64"), int(TensorProto.INT64), "TensorProto.INT64"
    ),
    int(TensorProto.BOOL): TensorDtypeMap(
        np.dtype("bool"), int(TensorProto.INT32), "TensorProto.BOOL"
    ),
    int(TensorProto.FLOAT16): TensorDtypeMap(
        np.dtype("float16"), int(TensorProto.UINT16), "TensorProto.FLOAT16"
    ),
    # Native numpy does not support bfloat16 so now use float32.
    int(TensorProto.BFLOAT16): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT16), "TensorProto.BFLOAT16"
    ),
    int(TensorProto.DOUBLE): TensorDtypeMap(
        np.dtype("float64"), int(TensorProto.DOUBLE), "TensorProto.DOUBLE"
    ),
    int(TensorProto.COMPLEX64): TensorDtypeMap(
        np.dtype("complex64"), int(TensorProto.FLOAT), "TensorProto.COMPLEX64"
    ),
    int(TensorProto.COMPLEX128): TensorDtypeMap(
        np.dtype("complex128"), int(TensorProto.DOUBLE), "TensorProto.COMPLEX128"
    ),
    int(TensorProto.UINT32): TensorDtypeMap(
        np.dtype("uint32"), int(TensorProto.UINT32), "TensorProto.UINT32"
    ),
    int(TensorProto.UINT64): TensorDtypeMap(
        np.dtype("uint64"), int(TensorProto.UINT64), "TensorProto.UINT64"
    ),
    int(TensorProto.STRING): TensorDtypeMap(
        np.dtype("object"), int(TensorProto.STRING), "TensorProto.STRING"
    ),
    # Native numpy does not support float8 types, so now use float32 for these types.
    # TODO(after #6605): Use ml_dtypes instead.
    int(TensorProto.FLOAT8E4M3FN): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E4M3FN"
    ),
    int(TensorProto.FLOAT8E4M3FNUZ): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E4M3FNUZ"
    ),
    int(TensorProto.FLOAT8E5M2): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E5M2"
    ),
    int(TensorProto.FLOAT8E5M2FNUZ): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E5M2FNUZ"
    ),
    # Native numpy does not support uint4/int4 so now use uint8/int8 for these types.
    int(TensorProto.UINT4): TensorDtypeMap(
        np.dtype("uint8"), int(TensorProto.INT32), "TensorProto.UINT4"
    ),
    int(TensorProto.INT4): TensorDtypeMap(
        np.dtype("int8"), int(TensorProto.INT32), "TensorProto.INT4"
    ),
    int(TensorProto.FLOAT4E2M1): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT4E2M1"
    ),
}
