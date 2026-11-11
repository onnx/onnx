# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from google.protobuf.descriptor import FieldDescriptor

from onnx import TensorProto

if TYPE_CHECKING:
    from google.protobuf.message import Message

# Keep this aligned with ProtoPrinter::print(const TensorProto&, bool).
_PRINTABLE_TYPES = {
    TensorProto.FLOAT,
    TensorProto.DOUBLE,
    TensorProto.FLOAT16,
    TensorProto.BFLOAT16,
    TensorProto.FLOAT8E4M3FN,
    TensorProto.FLOAT8E4M3FNUZ,
    TensorProto.FLOAT8E5M2,
    TensorProto.FLOAT8E5M2FNUZ,
    TensorProto.FLOAT8E8M0,
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
    TensorProto.BOOL,
    TensorProto.STRING,
}


def supports_text_round_trip(message: Message) -> bool:
    """Exclude tensor representations that the printer replaces with `...`."""
    if isinstance(message, TensorProto):
        # External tensors print their metadata rather than decoding data.
        if message.data_location == TensorProto.EXTERNAL:
            return True
        return message.data_type in _PRINTABLE_TYPES and not (
            message.data_type == TensorProto.STRING and message.HasField("raw_data")
        )
    for field, value in message.ListFields():
        if field.type != FieldDescriptor.TYPE_MESSAGE:
            continue
        is_repeated = getattr(field, "is_repeated", None)
        if is_repeated is None:
            is_repeated = field.label == FieldDescriptor.LABEL_REPEATED
        values = value if is_repeated else [value]
        if any(not supports_text_round_trip(item) for item in values):
            return False
    return True
