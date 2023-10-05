"""Protocols derived from onnx/onnx.proto3"""

# TODO: Not sure what I should use this for. We may want to structure
# the implementation differently.
# This protocols can be used for serialization, but we still need
# the implementer for traversal. Then we have
# IR objects -> this (plain) data structure -> serialization

from onnx.ir import _enums
from typing import Protocol, Any

import typing


@typing.runtime_checkable
class AttributeProtocol(Protocol):
    """Protocol for ONNX attributes."""

    name: str
    type: _enums.AttributeType
    value: Any
    ref_attr_name: str
    doc_string: str


class SegmentProtocol(Protocol):
    begin: int
    end: int


@typing.runtime_checkable
class TensorProtocol(Protocol):
    # This is a concrete tensor value. The dims are for interpreting the data
    dims: typing.Sequence[int]
    data_type: _enums.DataType
    name: str
    doc_string: str
    data: Any
    segment: typing.Optional[SegmentProtocol]
    external_data: typing.Sequence[typing.Mapping[str, str]]
    data_location: _enums.TensorDataLocation


@typing.runtime_checkable
class SparseTensorProtocol(Protocol):
    values: TensorProtocol
    indices: TensorProtocol
    dims: typing.Sequence[int]


# TODO: For symbolic shapes, the value may be an object dependent
# on other shapes. Potentially allow replacing the dimension implementation?
@typing.runtime_checkable
class DimensionProtocol(Protocol):
    value: int | str
    denotation: str


@typing.runtime_checkable
class ShapeProtocol(Protocol):
    # TODO: Integrate Dimension Denotation
    # https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition

    dims: typing.Sequence[DimensionProtocol]


@typing.runtime_checkable
class TypeProtocol(Protocol):
    denotation: str


@typing.runtime_checkable
class TensorTypeProtocol(TypeProtocol, Protocol):
    # NOTE: This is TypeProto.Tensor
    elem_type: _enums.DataType
    shape: ShapeProtocol


@typing.runtime_checkable
class SequenceTypeProtocol(TypeProtocol, Protocol):
    elem_type: TypeProtocol


@typing.runtime_checkable
class MapTypeProtocol(TypeProtocol, Protocol):
    key_type: typing.Literal[
        _enums.DataType.STRING,
        _enums.DataType.INT64,
        _enums.DataType.INT32,
        _enums.DataType.INT16,
        _enums.DataType.INT8,
        _enums.DataType.UINT64,
        _enums.DataType.UINT32,
        _enums.DataType.UINT16,
        _enums.DataType.UINT8,
    ]
    value_type: TypeProtocol


@typing.runtime_checkable
class OptionalTypeProtocol(TypeProtocol, Protocol):
    elem_type: TypeProtocol


@typing.runtime_checkable
class SparseTensorTypeProtocol(TypeProtocol, Protocol):
    # NOTE: This is TypeProto.SparseTensor
    elem_type: _enums.DataType
    shape: ShapeProtocol
