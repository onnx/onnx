# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import google.protobuf.message
import pytest

import onnx
from onnx import checker, helper, onnx_operators_pb


@pytest.mark.parametrize(
    ("proto_type", "field", "wire"),
    [
        (onnx.AttributeProto, "i", b"\x18\x00"),
        (onnx.AttributeProto, "s", b"\x22\x00"),
        (onnx.TensorProto, "raw_data", b"\x4a\x00"),
        (onnx.OperatorSetIdProto, "domain", b"\x0a\x00"),
    ],
)
def test_proto2_explicit_defaults_preserve_presence(proto_type, field, wire):
    # Historical proto2 encodings must keep explicit zero/empty values distinct
    # from an absent field, including after another serialization.
    proto = proto_type()
    assert not proto.HasField(field)
    proto.ParseFromString(wire)
    assert proto.HasField(field)
    assert proto.SerializeToString() == wire
    proto.ClearField(field)
    assert not proto.HasField(field)
    assert proto.SerializeToString() == b""


def test_proto2_tensor_encoding_is_preserved():
    # Unpacked dims [2] and packed int64_data [2, 3], as emitted by proto2.
    wire = b"\x08\x02\x10\x07\x3a\x02\x02\x03"
    tensor = onnx.TensorProto.FromString(wire)
    assert tensor.dims == [2]
    assert tensor.int64_data == [2, 3]
    assert tensor.SerializeToString() == wire


def test_bytes_fields_remain_binary():
    data = b"\xff\xfe\x00"
    tensor = onnx.TensorProto(raw_data=data)
    assert onnx.TensorProto.FromString(tensor.SerializeToString()).raw_data == data


@pytest.mark.parametrize(
    ("proto_type", "field", "wire"),
    [
        (onnx.AttributeProto, "type", b"\xa0\x01\x7f"),
        (onnx.TensorProto, "data_location", b"\x70\x7f"),
        (onnx_operators_pb.OperatorProto, "status", b"\x18\x7f"),
    ],
)
def test_unknown_enum_values_preserve_proto2_behavior(proto_type, field, wire):
    proto = proto_type.FromString(wire)
    assert not proto.HasField(field)
    assert getattr(proto, field) == 0
    assert proto.SerializeToString() == wire
    with pytest.raises(ValueError):
        setattr(proto, field, 127)
    proto.DiscardUnknownFields()
    assert proto.SerializeToString() == b""


def test_explicit_presence_does_not_add_synthetic_oneofs():
    # Reflection consumers must still distinguish scalar fields from real oneofs.
    for proto_type, field in [
        (onnx.AttributeProto, "i"),
        (onnx.TensorProto, "data_location"),
        (onnx_operators_pb.OperatorProto, "status"),
    ]:
        descriptor = proto_type.DESCRIPTOR.fields_by_name[field]
        assert descriptor.has_presence
        assert descriptor.containing_oneof is None
    dimension = onnx.TensorShapeProto.Dimension(dim_value=0)
    assert dimension.WhichOneof("value") == "dim_value"
    dimension.dim_param = "N"
    assert dimension.WhichOneof("value") == "dim_param"
    assert not dimension.HasField("dim_value")


@pytest.mark.parametrize("field", ["graph_name", "doc_string", "metadata"])
def test_invalid_utf8_is_rejected_by_python_and_cpp(field, tmp_path):
    marker = "utf8-marker"
    graph = helper.make_graph([], "圖形\U0001f600", [], [])
    model = helper.make_model(graph)
    model.doc_string = "說明"
    helper.set_model_props(model, {"作者": "測試"})
    if field == "graph_name":
        model.graph.name = marker
    elif field == "doc_string":
        model.doc_string = marker
    else:
        helper.set_model_props(model, {"key": marker})
    checker.check_model(model)
    wire = model.SerializeToString()
    assert onnx.load_model_from_string(wire) == model
    assert wire.count(marker.encode()) == 1
    replacement = b"\xff" + b"x" * (len(marker) - 1)
    corrupted = wire.replace(marker.encode(), replacement)

    # The pure-Python protobuf backend (used by free-threaded builds) raises
    # UnicodeDecodeError, while upb wraps it in DecodeError.
    with pytest.raises((google.protobuf.message.DecodeError, UnicodeDecodeError)):
        onnx.load_model_from_string(corrupted)
    # Pass bytes directly so this exercises the C++ parser, not Python's.
    with pytest.raises(ValueError, match="Unable to parse proto"):
        checker.check_model(corrupted)
    path = tmp_path / "invalid.onnx"
    path.write_bytes(corrupted)
    with pytest.raises(checker.ValidationError, match="Unable to parse proto"):
        checker.check_model(path)
