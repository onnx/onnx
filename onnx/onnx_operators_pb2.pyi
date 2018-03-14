# @generated by generate_proto_mypy_stubs.py.  Do not edit!
from google.protobuf.message import (
    Message,
)

from typing import (
    Iterable,
    List,
    Tuple,
    cast,
    Text,
)

from mypy import (
    RepeatedScalarFieldContainer,
)

class OperatorProto(Message):
    class OperatorStatus(int):
        @classmethod
        def Name(cls, number: int) -> str: ...
        @classmethod
        def Value(cls, name: str) -> int: ...
        @classmethod
        def keys(cls) -> List[str]: ...
        @classmethod
        def values(cls) -> List[int]: ...
        @classmethod
        def items(cls) -> List[Tuple[str, int]]: ...
    EXPERIMENTAL = cast(OperatorStatus, 0)
    STABLE = cast(OperatorStatus, 1)
    
    op_type = ... # type: Text
    since_version = ... # type: int
    status = ... # type: OperatorProto.OperatorStatus
    doc_string = ... # type: Text
    
    def __init__(self,
        op_type : Text = None,
        since_version : int = None,
        status : OperatorProto.OperatorStatus = None,
        doc_string : Text = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> OperatorProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class OperatorSetProto(Message):
    magic = ... # type: Text
    ir_version = ... # type: int
    ir_version_prerelease = ... # type: Text
    ir_build_metadata = ... # type: Text
    domain = ... # type: Text
    opset_version = ... # type: int
    doc_string = ... # type: Text
    
    @property
    def operator(self) -> RepeatedScalarFieldContainer[OperatorProto]: ...
    
    def __init__(self,
        magic : Text = None,
        ir_version : int = None,
        ir_version_prerelease : Text = None,
        ir_build_metadata : Text = None,
        domain : Text = None,
        opset_version : int = None,
        doc_string : Text = None,
        operator : Iterable[OperatorProto] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> OperatorSetProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...
