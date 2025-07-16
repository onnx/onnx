from onnx import onnx_ml_pb2 as _onnx_ml_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OperatorProto(_message.Message):
    __slots__ = ("op_type", "since_version", "status", "doc_string")
    OP_TYPE_FIELD_NUMBER: _ClassVar[int]
    SINCE_VERSION_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    op_type: str
    since_version: int
    status: _onnx_ml_pb2.OperatorStatus
    doc_string: str
    def __init__(self, op_type: _Optional[str] = ..., since_version: _Optional[int] = ..., status: _Optional[_Union[_onnx_ml_pb2.OperatorStatus, str]] = ..., doc_string: _Optional[str] = ...) -> None: ...

class OperatorSetProto(_message.Message):
    __slots__ = ("magic", "ir_version", "ir_version_prerelease", "ir_build_metadata", "domain", "opset_version", "doc_string", "operator", "functions")
    MAGIC_FIELD_NUMBER: _ClassVar[int]
    IR_VERSION_FIELD_NUMBER: _ClassVar[int]
    IR_VERSION_PRERELEASE_FIELD_NUMBER: _ClassVar[int]
    IR_BUILD_METADATA_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    OPSET_VERSION_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    OPERATOR_FIELD_NUMBER: _ClassVar[int]
    FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    magic: str
    ir_version: int
    ir_version_prerelease: str
    ir_build_metadata: str
    domain: str
    opset_version: int
    doc_string: str
    operator: _containers.RepeatedCompositeFieldContainer[OperatorProto]
    functions: _containers.RepeatedCompositeFieldContainer[_onnx_ml_pb2.FunctionProto]
    def __init__(self, magic: _Optional[str] = ..., ir_version: _Optional[int] = ..., ir_version_prerelease: _Optional[str] = ..., ir_build_metadata: _Optional[str] = ..., domain: _Optional[str] = ..., opset_version: _Optional[int] = ..., doc_string: _Optional[str] = ..., operator: _Optional[_Iterable[_Union[OperatorProto, _Mapping]]] = ..., functions: _Optional[_Iterable[_Union[_onnx_ml_pb2.FunctionProto, _Mapping]]] = ...) -> None: ...
