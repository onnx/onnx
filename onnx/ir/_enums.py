import enum

class AttributeType(enum.IntEnum):
    """Enum for the types of ONNX attributes."""

    # NOTE: Naming follows python conventions.
    # C++ names can follow C++ conventions and rename when binding.

    # TODO: Should we code gen this? We just need to get rid of protoc
    # We can code gen with https://github.com/recap-build/proto-schema-parser/tree/main

    # NOTE: We can assume the build tool chain has python, just not protoc, right?
    # NOTE: We should alias OpSchema::AttrType as well
    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSOR = 11
    SPARSE_TENSORS = 12
    TYPE_PROTO = 13
    TYPE_PROTOS = 14


class DataType(enum.IntEnum):
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20


class TensorDataLocation(enum.IntEnum):
    DEFAULT = 0
    EXTERNAL = 1
