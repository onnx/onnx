# SPDX-License-Identifier: Apache-2.0

from onnx import TensorProto, SequenceProto, OptionalProto
import numpy as np  # type: ignore

# This map is used for converting TensorProto values into Numpy arrays
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
    int(TensorProto.BFLOAT16): np.dtype('float32'),  # Native numpy does not support bfloat16 so now use float32 for bf16 values
    int(TensorProto.DOUBLE): np.dtype('float64'),
    int(TensorProto.COMPLEX64): np.dtype('complex64'),
    int(TensorProto.COMPLEX128): np.dtype('complex128'),
    int(TensorProto.UINT32): np.dtype('uint32'),
    int(TensorProto.UINT64): np.dtype('uint64'),
    int(TensorProto.STRING): np.dtype('object')
}

# Currently native numpy does not support bfloat16 so TensorProto.BFLOAT16 is ignored for now
# Numpy float32 array is only reversed to TensorProto.FLOAT
NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items() if k != TensorProto.BFLOAT16}

# This is only used to get keys into STORAGE_TENSOR_TYPE_TO_FIELD.
# TODO(https://github.com/onnx/onnx/issues/4261): Remove this.
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

STORAGE_ELEMENT_TYPE_TO_FIELD = {
    int(SequenceProto.TENSOR): 'tensor_values',
    int(SequenceProto.SPARSE_TENSOR): 'sparse_tensor_values',
    int(SequenceProto.SEQUENCE): 'sequence_values',
    int(SequenceProto.MAP): 'map_values',
    int(OptionalProto.OPTIONAL): 'optional_value'
}

OPTIONAL_ELEMENT_TYPE_TO_FIELD = {
    int(OptionalProto.TENSOR): 'tensor_value',
    int(OptionalProto.SPARSE_TENSOR): 'sparse_tensor_value',
    int(OptionalProto.SEQUENCE): 'sequence_value',
    int(OptionalProto.MAP): 'map_value',
    int(OptionalProto.OPTIONAL): 'optional_value'
}
