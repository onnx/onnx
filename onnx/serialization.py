# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

__all__ = [
    "registry",
]

import typing
from typing import Any, Collection, Optional, Protocol, TypeVar

import google.protobuf.message
import google.protobuf.text_format

import onnx

_Proto = TypeVar("_Proto", bound=google.protobuf.message.Message)
# Encoding used for serializing and deserializing text files
_ENCODING = "utf-8"


class ProtoSerializer(Protocol):
    """A serializer-deserializer to and from in-memory Protocol Buffers representations."""

    supported_formats: Collection[str]

    # NOTE: The methods defined are serialize_proto and deserialize_proto and not the
    # more generic serialize and deserialize to leave space for future protocols
    # that are defined to serialize/deserialize the ONNX in memory IR.
    # This way a class can implement both protocols.

    def serialize_proto(self, proto: _Proto) -> Any:
        """Serialize a in-memory proto to a serialized data type."""

    def deserialize_proto(self, serialized: Any, proto: _Proto) -> _Proto:
        """Parse a serialized data type into a in-memory proto."""


class _Registry:
    def __init__(self) -> None:
        self._serializers: dict[str, ProtoSerializer] = {}

    def register(self, serializer: ProtoSerializer) -> None:
        for fmt in serializer.supported_formats:
            self._serializers[fmt] = serializer

    def get(self, fmt: str) -> ProtoSerializer:
        """Get a serializer for a format.

        Args:
            fmt (str): The format to get a serializer for.

        Returns:
            ProtoSerializer: The serializer for the format.

        Raises:
            ValueError: If the format is not supported.
        """
        try:
            return self._serializers[fmt]
        except KeyError:
            raise ValueError(
                f"Unsupported format: '{fmt}'. Supported formats are: {self._serializers.keys()}"
            ) from None


class _ProtobufSerializer(ProtoSerializer):
    """Serialize and deserialize protobuf message."""

    supported_formats = ("protobuf",)

    def serialize_proto(self, proto: _Proto) -> bytes:
        if hasattr(proto, "SerializeToString") and callable(proto.SerializeToString):
            try:
                result = proto.SerializeToString()
            except ValueError as e:
                if proto.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
                    raise ValueError(
                        "The proto size is larger than the 2 GB limit. "
                        "Please use save_as_external_data to save tensors separately from the model file."
                    ) from e
                raise
            return result  # type: ignore
        raise TypeError(
            f"No SerializeToString method is detected.\ntype is {type(proto)}"
        )

    def deserialize_proto(self, serialized: bytes, proto: _Proto) -> _Proto:
        if not isinstance(serialized, bytes):
            raise TypeError(
                f"Parameter 'serialized' must be bytes, but got type: {type(serialized)}"
            )
        decoded = typing.cast(Optional[int], proto.ParseFromString(serialized))
        if decoded is not None and decoded != len(serialized):
            raise google.protobuf.message.DecodeError(
                f"Protobuf decoding consumed too few bytes: {decoded} out of {len(serialized)}"
            )
        return proto


class _TextProtoSerializer(ProtoSerializer):
    """Serialize and deserialize text proto."""

    supported_formats = ("textproto",)

    def serialize_proto(self, proto: _Proto) -> bytes:
        textproto = google.protobuf.text_format.MessageToString(proto)
        return textproto.encode(_ENCODING)

    def deserialize_proto(self, serialized: bytes | str, proto: _Proto) -> _Proto:
        if not isinstance(serialized, (bytes, str)):
            raise TypeError(
                f"Parameter 'serialized' must be bytes or str, but got type: {type(serialized)}"
            )
        if isinstance(serialized, bytes):
            serialized = serialized.decode(_ENCODING)
        assert isinstance(serialized, str)
        return google.protobuf.text_format.Parse(serialized, proto)


# Register default serializers
registry = _Registry()
registry.register(_ProtobufSerializer())
registry.register(_TextProtoSerializer())
