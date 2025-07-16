from onnx import onnx_ml_pb2 as _onnx_ml_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SequenceProto(_message.Message):
    __slots__ = ("name", "elem_type", "tensor_values", "sparse_tensor_values", "sequence_values", "map_values", "optional_values")
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNDEFINED: _ClassVar[SequenceProto.DataType]
        TENSOR: _ClassVar[SequenceProto.DataType]
        SPARSE_TENSOR: _ClassVar[SequenceProto.DataType]
        SEQUENCE: _ClassVar[SequenceProto.DataType]
        MAP: _ClassVar[SequenceProto.DataType]
        OPTIONAL: _ClassVar[SequenceProto.DataType]
    UNDEFINED: SequenceProto.DataType
    TENSOR: SequenceProto.DataType
    SPARSE_TENSOR: SequenceProto.DataType
    SEQUENCE: SequenceProto.DataType
    MAP: SequenceProto.DataType
    OPTIONAL: SequenceProto.DataType
    NAME_FIELD_NUMBER: _ClassVar[int]
    ELEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_VALUES_FIELD_NUMBER: _ClassVar[int]
    SPARSE_TENSOR_VALUES_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_VALUES_FIELD_NUMBER: _ClassVar[int]
    MAP_VALUES_FIELD_NUMBER: _ClassVar[int]
    OPTIONAL_VALUES_FIELD_NUMBER: _ClassVar[int]
    name: str
    elem_type: int
    tensor_values: _containers.RepeatedCompositeFieldContainer[_onnx_ml_pb2.TensorProto]
    sparse_tensor_values: _containers.RepeatedCompositeFieldContainer[_onnx_ml_pb2.SparseTensorProto]
    sequence_values: _containers.RepeatedCompositeFieldContainer[SequenceProto]
    map_values: _containers.RepeatedCompositeFieldContainer[MapProto]
    optional_values: _containers.RepeatedCompositeFieldContainer[OptionalProto]
    def __init__(self, name: _Optional[str] = ..., elem_type: _Optional[int] = ..., tensor_values: _Optional[_Iterable[_Union[_onnx_ml_pb2.TensorProto, _Mapping]]] = ..., sparse_tensor_values: _Optional[_Iterable[_Union[_onnx_ml_pb2.SparseTensorProto, _Mapping]]] = ..., sequence_values: _Optional[_Iterable[_Union[SequenceProto, _Mapping]]] = ..., map_values: _Optional[_Iterable[_Union[MapProto, _Mapping]]] = ..., optional_values: _Optional[_Iterable[_Union[OptionalProto, _Mapping]]] = ...) -> None: ...

class MapProto(_message.Message):
    __slots__ = ("name", "key_type", "keys", "string_keys", "values")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_TYPE_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    STRING_KEYS_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    name: str
    key_type: int
    keys: _containers.RepeatedScalarFieldContainer[int]
    string_keys: _containers.RepeatedScalarFieldContainer[bytes]
    values: SequenceProto
    def __init__(self, name: _Optional[str] = ..., key_type: _Optional[int] = ..., keys: _Optional[_Iterable[int]] = ..., string_keys: _Optional[_Iterable[bytes]] = ..., values: _Optional[_Union[SequenceProto, _Mapping]] = ...) -> None: ...

class OptionalProto(_message.Message):
    __slots__ = ("name", "elem_type", "tensor_value", "sparse_tensor_value", "sequence_value", "map_value", "optional_value")
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNDEFINED: _ClassVar[OptionalProto.DataType]
        TENSOR: _ClassVar[OptionalProto.DataType]
        SPARSE_TENSOR: _ClassVar[OptionalProto.DataType]
        SEQUENCE: _ClassVar[OptionalProto.DataType]
        MAP: _ClassVar[OptionalProto.DataType]
        OPTIONAL: _ClassVar[OptionalProto.DataType]
    UNDEFINED: OptionalProto.DataType
    TENSOR: OptionalProto.DataType
    SPARSE_TENSOR: OptionalProto.DataType
    SEQUENCE: OptionalProto.DataType
    MAP: OptionalProto.DataType
    OPTIONAL: OptionalProto.DataType
    NAME_FIELD_NUMBER: _ClassVar[int]
    ELEM_TYPE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_VALUE_FIELD_NUMBER: _ClassVar[int]
    SPARSE_TENSOR_VALUE_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_VALUE_FIELD_NUMBER: _ClassVar[int]
    MAP_VALUE_FIELD_NUMBER: _ClassVar[int]
    OPTIONAL_VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    elem_type: int
    tensor_value: _onnx_ml_pb2.TensorProto
    sparse_tensor_value: _onnx_ml_pb2.SparseTensorProto
    sequence_value: SequenceProto
    map_value: MapProto
    optional_value: OptionalProto
    def __init__(self, name: _Optional[str] = ..., elem_type: _Optional[int] = ..., tensor_value: _Optional[_Union[_onnx_ml_pb2.TensorProto, _Mapping]] = ..., sparse_tensor_value: _Optional[_Union[_onnx_ml_pb2.SparseTensorProto, _Mapping]] = ..., sequence_value: _Optional[_Union[SequenceProto, _Mapping]] = ..., map_value: _Optional[_Union[MapProto, _Mapping]] = ..., optional_value: _Optional[_Union[OptionalProto, _Mapping]] = ...) -> None: ...
