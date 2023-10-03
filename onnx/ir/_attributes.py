from __future__ import annotations

import enum
from typing import (
    Any,
    Generic,
    Protocol,
    Sequence,
    TypeVar,
)



S = TypeVar("S")
T = TypeVar("T")
AttrT = TypeVar("AttrT", bound="Attribute")
AttrIterableT = TypeVar("AttrIterableT", bound="_AttrIterable")


class AttributeType(enum.IntEnum):
    """Enum for the types of ONNX attributes."""

    # NOTE: Naming follows python conventions.
    # C++ names can follow C++ conventions and rename when binding.

    # TODO: Should we code gen this? We just need to get rid of protoc
    # We can code gen with https://github.com/recap-build/proto-schema-parser/tree/main

    # NOTE: We can assume the build tool chain has python, just not protoc, right?

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


# NOTE: None of these classes will have a "to_onnx" method because
# We cannot assume that the build tool chain has protoc installed.

class AttributeProtocol(Protocol):
    """Protocol for ONNX attributes."""

    name: str
    type: AttributeType
    value: Any
    ref_attr_name: str
    doc_string: str


class Attribute(Generic[T], AttributeProtocol):
    """Base class for ONNX attributes."""

    # NOTE: We use primitive types for T

    def __init__(self,
        name: str,
        type: AttributeType,
        value: T,
        *,
        ref_attr_name: str,
        doc_string: str
    ):
        self.name = name
        self.type = type
        self.value = value
        self.ref_attr_name = ref_attr_name
        self.doc_string = doc_string

    # TODO: How do we represent ref attributes? Do we need to?


# NOTE: The following classes are just supporting classes (partially applied) for convenience
# But I think they would be useful to have in the IR by having the type info
# explicitly in the class type.
# Arguably, they can also be functions that return Attribute objects.
# TODO: We need something that is easy for pattern matchers, otherwise they
# can just be make_attribute functions.
class AttrFloat32(Attribute[float]):
    def __init__(self, name: str, value: float, ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.FLOAT, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrInt64(Attribute[int]):
    def __init__(self, name: str, value: int, ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.INT, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrString(Attribute[str]):
    def __init__(self, name: str, value: str, ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.STRING, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

# NOTE: Tensor should be a tensor proto
class AttrTensor(Attribute[Tensor]):
    def __init__(self, name: str, value: Tensor, ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.TENSOR, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrGraph(Attribute[Graph]):
    def __init__(self, name: str, value: Graph, ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.GRAPH, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrFloat32s(Attribute[Sequence[float]]):
    def __init__(self, name: str, value: Sequence[float], ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.FLOATS, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrInt64s(Attribute[Sequence[int]]):
    def __init__(self, name: str, value: Sequence[int], ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.INTS, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrStrings(Attribute[Sequence[str]]):
    def __init__(self, name: str, value: Sequence[str], ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.STRINGS, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrTensors(Attribute[Sequence[Tensor]]):
    def __init__(self, name: str, value: Sequence[Tensor], ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.TENSORS, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrGraphs(Attribute[Sequence[Graph]]):
    def __init__(self, name: str, value: Sequence[Graph], ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.GRAPHS, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

# NOTE: SparseTensor should be a sparse tensor proto
class AttrSparseTensor(Attribute[Sequence[SparseTensor]]):
    def __init__(self, name: str, value: Sequence[SparseTensor], ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.SPARSE_TENSOR, value, ref_attr_name=ref_attr_name, doc_string=doc_string)

class AttrSparseTensors(Attribute[Sequence[SparseTensor]]):
    def __init__(self, name: str, value: Sequence[SparseTensor], ref_attr_name: str = "", doc_string: str = ""):
        super().__init__(name, AttributeType.SPARSE_TENSORS, value, ref_attr_name=ref_attr_name, doc_string=doc_string)


# class AttrTypeProto(Attribute[Sequence[TypeProto]]):
#     def __init__(self, name: str, value: Sequence[TypeProto], ref_attr_name: str = "", doc_string: str = ""):
#         super().__init__(name, AttributeType.TYPE_PROTO, value, ref_attr_name=ref_attr_name, doc_string=doc_string)
