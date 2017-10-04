from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
from onnx.onnx_pb2 import TensorProto


if sys.byteorder != 'little':
    raise RuntimeError(
        'Numpy helper for tensor/ndarray is not available on big endian '
        'systems yet.')

def combine_pairs_to_complex(fa)
    return [complex(fa[i * 2], fa[i * 2 + 1]) for i in range(len(fa)//2)]

def to_array(tensor):
    """Converts a tensor def object to a numpy array.

    Inputs:
        tensor: a TensorProto object.
    Returns:
        arr: the converted array.
    """
    if tensor.HasField("segment"):
        raise ValueError(
            "Currently not supporting loading segments.")
    if tensor.data_type == TensorProto.UNDEFINED:
        raise ValueError("The data type is not defined.")
    data_type = tensor.data_type
    dims = tensor.dims
    if tensor.HasField("raw_data"):
        # Raw_bytes support: using frombuffer.
        dtype_map = {
            TensorProto.FLOAT: np.float32,
            TensorProto.UINT8: np.uint8,
            TensorProto.INT8: np.int8,
            TensorProto.UINT16: np.uint16,
            TensorProto.INT16: np.int16,
            TensorProto.INT32: np.int32,
            TensorProto.INT64: np.int64,
            TensorProto.BOOL: np.bool,
            TensorProto.FLOAT16: np.float16,
            TensorProto.DOUBLE: np.float64,
            TensorProto.UINT32: np.uint32,
            TensorProto.UINT64: np.uint64,
            TensorProto.COMPLEX64: np.complex64,
            TensorProto.COMPLEX128: np.complex128,
        }
        if data_type == TensorProto.STRING:
            raise RuntimeError("You cannot use raw bytes for string type.")
        try:
            dtype = dtype_map[tensor.data_type]
        except KeyError:
            # TODO: complete the data type.
            raise RuntimeError(
                "Tensor data type not understood yet: {}".format(str(data_type)))
        return np.frombuffer(
            tensor.raw_data,
            dtype=dtype).reshape(dims)
    else:
        # Conventional fields not using raw bytes.
        if data_type == TensorProto.FLOAT:
            return np.asarray(
                tensor.float_data, dtype=np.float32).reshape(dims)
        elif data_type == TensorProto.COMPLEX64:
            return np.asarray(
                combine_pairs_to_complex(tensor.float_data), dtype=np.complex64).reshape(dims)
        elif data_type == TensorProto.UINT8:
            return np.asarray(
                tensor.int32_data, dtype=np.int32).reshape(dims).astype(np.uint8)
        elif data_type == TensorProto.INT8:
            return np.asarray(
                tensor.int32_data, dtype=np.int32).reshape(dims).astype(np.int8)
        elif data_type == TensorProto.UINT16:
            return np.asarray(
                tensor.int32_data, dtype=np.int32).reshape(dims).astype(np.uint16)
        elif data_type == TensorProto.INT16:
            return np.asarray(
                tensor.int32_data, dtype=np.int32).reshape(dims).astype(np.int16)
        elif data_type == TensorProto.INT32:
            return np.asarray(
                tensor.int32_data, dtype=np.int32).reshape(dims)
        elif data_type == TensorProto.INT64:
            return np.asarray(
                tensor.int64_data, dtype=np.int64).reshape(dims)
        elif data_type == TensorProto.UINT32:
            return np.asarray(
                tensor.uint64_data, dtype=np.uint32).reshape(dims)
        elif data_type == TensorProto.UINT64:
            return np.asarray(
                tensor.uint64_data, dtype=np.uint64).reshape(dims)
        elif data_type == TensorProto.DOUBLE:
            return np.asarray(
                tensor.double_data, dtype=np.float64).reshape(dims)
        elif data_type == TensorProto.COMPLEX128:
            return np.asarray(
                combine_pairs_to_complex(tensor.double_data), dtype=np.complex128).reshape(dims)
        elif data_type == TensorProto.STRING:
            raise NotImplementedError("Not implemented.")
        elif data_type == TensorProto.BOOL:
            return np.asarray(
                tensor.int32_data, dtype=np.int32).reshape(dims).astype(np.bool)
        elif data_type == TensorProto.FLOAT16:
            return np.asarray(
                tensor.int32_data, dtype=np.uint16).reshape(dims).view(np.float16)
        else:
            # TODO: complete the data type.
            raise RuntimeError(
                "Tensor data type not understood yet: {}".format(str(data_type)))


def from_array(arr, name=None):
    """Converts a numpy array to a tensor def.

    Inputs:
        arr: a numpy array.
        name: (optional) the name of the tensor.
    Returns:
        tensor_def: the converted tensor def.
    """
    tensor = TensorProto()
    tensor.dims.extend(arr.shape)
    if name:
        tensor.name = name

    dtype_map = {
        np.dtype("float32"): TensorProto.FLOAT,
        np.dtype("uint8"): TensorProto.UINT8,
        np.dtype("int8"): TensorProto.INT8,
        np.dtype("uint16"): TensorProto.UINT16,
        np.dtype("int16"): TensorProto.INT16,
        np.dtype("int32"): TensorProto.INT32,
        np.dtype("int64"): TensorProto.INT64,
        np.dtype("bool"): TensorProto.BOOL,
        np.dtype("float16"): TensorProto.FLOAT16,
        np.dtype("float64"): TensorProto.DOUBLE,
        np.dtype("complex64"): TensorProto.COMPLEX64,
        np.dtype("complex128"): TensorProto.COMPLEX128,
        np.dtype("uint32"): TensorProto.UINT32,
        np.dtype("uint64"): TensorProto.UINT64,
        
    }

    if arr.dtype == np.object:
        # Special care for strings.
        raise NotImplementedError("Need to properly implement string.")
    # For numerical types, directly use numpy raw bytes.
    try:
        dtype = dtype_map[arr.dtype]
    except KeyError:
        raise RuntimeError(
            "Numpy data type not understood yet: {}".format(str(arr.dtype)))
    tensor.data_type = dtype
    tensor.raw_data = arr.tobytes()  # note: tobytes() is only after 1.9.
    
    return tensor
