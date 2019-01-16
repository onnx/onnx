from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import TensorProto
from typing import Text, Any
import numpy as np  # type: ignore

TENSOR_TYPE_TO_NP_TYPE = {
    int(TensorProto.FLOAT): np.dtype('float32'),
    int(TensorProto.UINT8): np.dtype('uint8'),
    int(TensorProto.INT8): np.dtype('int8'),
    int(TensorProto.UINT16): np.dtype('uint16'),
    int(TensorProto.INT16): np.dtype('int16'),
    int(TensorProto.INT32): np.dtype('int32'),
    int(TensorProto.INT64): np.dtype('int64'),
    int(TensorProto.BOOL): np.dtype('bool'),
    int(TensorProto.FLOAT16): np.dtype('float16'),
    int(TensorProto.DOUBLE): np.dtype('float64'),
    int(TensorProto.COMPLEX64): np.dtype('complex64'),
    int(TensorProto.COMPLEX128): np.dtype('complex128'),
    int(TensorProto.UINT32): np.dtype('uint32'),
    int(TensorProto.UINT64): np.dtype('uint64'),
    int(TensorProto.STRING): np.dtype(np.object)
}

NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items()}

TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = {
    int(TensorProto.FLOAT): int(TensorProto.FLOAT),
    int(TensorProto.UINT8): int(TensorProto.INT32),
    int(TensorProto.INT8): int(TensorProto.INT32),
    int(TensorProto.UINT16): int(TensorProto.INT32),
    int(TensorProto.INT16): int(TensorProto.INT32),
    int(TensorProto.INT32): int(TensorProto.INT32),
    int(TensorProto.INT64): int(TensorProto.INT64),
    int(TensorProto.BOOL): int(TensorProto.INT32),
    int(TensorProto.FLOAT16): int(TensorProto.UINT16),
    int(TensorProto.BFLOAT16): int(TensorProto.UINT16),
    int(TensorProto.DOUBLE): int(TensorProto.DOUBLE),
    int(TensorProto.COMPLEX64): int(TensorProto.FLOAT),
    int(TensorProto.COMPLEX128): int(TensorProto.DOUBLE),
    int(TensorProto.UINT32): int(TensorProto.UINT32),
    int(TensorProto.UINT64): int(TensorProto.UINT64),
    int(TensorProto.STRING): int(TensorProto.STRING),
}

STORAGE_TENSOR_TYPE_TO_FIELD = {
    int(TensorProto.FLOAT): 'float_data',
    int(TensorProto.INT32): 'int32_data',
    int(TensorProto.INT64): 'int64_data',
    int(TensorProto.UINT16): 'int32_data',
    int(TensorProto.DOUBLE): 'double_data',
    int(TensorProto.COMPLEX64): 'float_data',
    int(TensorProto.COMPLEX128): 'double_data',
    int(TensorProto.UINT32): 'uint64_data',
    int(TensorProto.UINT64): 'uint64_data',
    int(TensorProto.STRING): 'string_data',
    int(TensorProto.BOOL): 'int32_data',
}
