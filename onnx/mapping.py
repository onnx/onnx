from onnx.onnx_pb2 import TensorProto
from numpy import np


NP_TO_TENSOR = {
    np.float32: TensorProto.FLOAT,
    np.uint8: TensorProto.UINT8,
    np.int8: TensorProto.INT8,
    np.uint16: TensorProto.UINT16,
    np.int16: TensorProto.INT16,
    np.int32: TensorProto.INT32,
    np.int64: TensorProto.INT64,
    np.bool: TensorProto.BOOL,
    np.float16: TensorProto.FLOAT16,
    np.float64: TensorProto.DOUBLE,
    np.complex64: TensorProto.COMPLEX64,
    np.complex128: TensorProto.COMPLEX128,
    np.uint32: TensorProto.UINT32,
    np.uint64: TensorProto.UINT64,
}

TENSOR_TO_NP = {v: k for k, v in NP_TO_TENSOR.items()}
