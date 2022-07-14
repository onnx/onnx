# SPDX-License-Identifier: Apache-2.0

from onnx import TensorProto, SequenceProto, OptionalProto
import numpy as np  # type: ignore
from typing import Any, Dict
import warnings

# tensor_type: (numpy type, storage type, string name)
TENSOR_TYPE_MAP = {
    int(TensorProto.FLOAT): (np.dtype('float32'), int(TensorProto.FLOAT), 'TensorProto.FLOAT'),
    int(TensorProto.UINT8): (np.dtype('uint8'), int(TensorProto.INT32), 'TensorProto.UINT8'),
    int(TensorProto.INT8): (np.dtype('int8'), int(TensorProto.INT32), 'TensorProto.INT8'),
    int(TensorProto.UINT16): (np.dtype('uint16'), int(TensorProto.INT32), 'TensorProto.UINT16'),
    int(TensorProto.INT16): (np.dtype('int16'), int(TensorProto.INT32), 'TensorProto.INT16'),
    int(TensorProto.INT32): (np.dtype('int32'), int(TensorProto.INT32), 'TensorProto.INT32'),
    int(TensorProto.INT64): (np.dtype('int64'), int(TensorProto.INT64), 'TensorProto.INT64'),
    int(TensorProto.BOOL): (np.dtype('bool'), int(TensorProto.INT32), 'TensorProto.BOOL'),
    int(TensorProto.FLOAT16): (np.dtype('float16'), int(TensorProto.UINT16), 'TensorProto.FLOAT16'),
    # Native numpy does not support bfloat16 so now use float32 for bf16 values
    # TODO ONNX should dirtectly use bfloat16 for bf16 values after numpy has supported bfloat16 type
    int(TensorProto.BFLOAT16): (np.dtype('float32'), int(TensorProto.UINT16), 'TensorProto.BFLOAT16'),
    int(TensorProto.DOUBLE): (np.dtype('float64'), int(TensorProto.DOUBLE), 'TensorProto.DOUBLE'),
    int(TensorProto.COMPLEX64): (np.dtype('complex64'), int(TensorProto.FLOAT), 'TensorProto.COMPLEX64'),
    int(TensorProto.COMPLEX128): (np.dtype('complex128'), int(TensorProto.DOUBLE), 'TensorProto.COMPLEX128'),
    int(TensorProto.UINT32): (np.dtype('uint32'), int(TensorProto.UINT32), 'TensorProto.UINT32'),
    int(TensorProto.UINT64): (np.dtype('uint64'), int(TensorProto.UINT64), 'TensorProto.UINT64'),
    int(TensorProto.STRING): (np.dtype('object'), int(TensorProto.STRING), 'TensorProto.STRING'),
}

# This map is used for converting TensorProto values into numpy arrays
TENSOR_TYPE_TO_NP_TYPE = {tensor_type: value[0] for tensor_type, value in TENSOR_TYPE_MAP.items()}
# This is only used to get keys into STORAGE_TENSOR_TYPE_TO_FIELD.
# TODO(https://github.com/onnx/onnx/issues/4261): Remove this.


class WarningDict(dict):  # type: ignore
    def __getitem__(self, key: str) -> Any:
        warnings.warn(str("`mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE` is deprecated in ONNX 1.13 and will "
            + "be removed in next release. To silence this warning, please use `to_storage_tensor_type` instead."), DeprecationWarning, stacklevel=2)
        return super().__getitem__(key)


TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = WarningDict({tensor_type: value[1] for tensor_type, value in TENSOR_TYPE_MAP.items()})

# Currently native numpy does not support bfloat16 so TensorProto.BFLOAT16 is ignored for now
# Numpy float32 array is only reversed to TensorProto.FLOAT
NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in TENSOR_TYPE_TO_NP_TYPE.items() if k != TensorProto.BFLOAT16}

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


def to_np_type(tensor_type: int) -> Any:
    return TENSOR_TYPE_MAP[int(tensor_type)][0]


def to_storage_tensor_type(tensor_type: int) -> int:
    return TENSOR_TYPE_MAP[tensor_type][1]


def to_string(tensor_type: int) -> str:
    return TENSOR_TYPE_MAP[int(tensor_type)][2]


def to_storage_numpy_type(tensor_type: int) -> Any:
    return to_np_type(to_storage_tensor_type(tensor_type))


# This map is used to get storage field for certain tensor type
def to_field(tensor_type: int) -> str:
    return STORAGE_TENSOR_TYPE_TO_FIELD[TENSOR_TYPE_MAP[int(tensor_type)][1]]
