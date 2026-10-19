# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import google.protobuf.message
import pytest

import onnx
from onnx import checker, helper


@pytest.mark.parametrize(
    ("proto_type", "field", "wire"),
    [
        (onnx.AttributeProto, "i", b"\x18\x00"),
        (onnx.AttributeProto, "f", b"\x15\x00\x00\x00\x00"),
        (onnx.AttributeProto, "s", b"\x22\x00"),
        (onnx.TensorProto, "raw_data", b"\x4a\x00"),
        (onnx.TensorProto, "data_location", b"\x70\x00"),
        (onnx.ModelProto, "model_version", b"\x28\x00"),
        (onnx.OperatorSetIdProto, "domain", b"\x0a\x00"),
        (onnx.FunctionProto, "domain", b"\x52\x00"),
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


@pytest.mark.parametrize(
    ("proto_type", "field", "wire"),
    [
        (onnx.TensorProto, "dims", b"\x08\x02\x08\x03"),
        (onnx.SparseTensorProto, "dims", b"\x18\x02\x18\x03"),
        (onnx.AttributeProto, "ints", b"\x40\x02\x40\x03"),
        (onnx.MapProto, "keys", b"\x18\x02\x18\x03"),
        (onnx.TensorProto, "int64_data", b"\x3a\x02\x02\x03"),
    ],
)
def test_proto2_repeated_numeric_encoding_is_preserved(proto_type, field, wire):
    proto = proto_type.FromString(wire)
    assert list(getattr(proto, field)) == [2, 3]
    assert proto.SerializeToString() == wire


def test_unknown_fields_survive_round_trip():
    # A future length-delimited field (number 100) remains opaque binary data.
    wire = b"\xa2\x06\x02\xff\xfe"
    model = onnx.ModelProto.FromString(wire)
    assert model.SerializeToString() == wire


def test_unknown_enum_values_are_preserved_as_values():
    # Proto3 enums are open: unknown values are exposed, unlike proto2's
    # unknown-field representation. This intentional change needs documenting.
    wire = b"\x70\x7f"
    unknown_value = 127
    tensor = onnx.TensorProto.FromString(wire)
    assert tensor.HasField("data_location")
    assert tensor.data_location == unknown_value
    assert tensor.SerializeToString() == wire


def test_bytes_fields_remain_binary():
    data = b"\xff\xfe\x00"
    attribute = onnx.AttributeProto(s=data)
    assert onnx.AttributeProto.FromString(attribute.SerializeToString()).s == data
    tensor = onnx.TensorProto(raw_data=data, string_data=[data])
    restored = onnx.TensorProto.FromString(tensor.SerializeToString())
    assert restored.raw_data == data
    assert restored.string_data == [data]


@pytest.mark.parametrize(
    "bad_utf8",
    [b"\xff", b"\x80", b"\xc0\x80", b"\xe4\xb8", b"\xed\xa0\x80", b"\xf4\x90\x80\x80"],
)
@pytest.mark.parametrize("field", ["graph_name", "doc_string", "metadata"])
def test_invalid_utf8_is_rejected_by_python_and_cpp(bad_utf8, field, tmp_path):
    marker = "utf8-marker"
    graph = helper.make_graph([], "graph", [], [])
    model = helper.make_model(graph)
    if field == "graph_name":
        model.graph.name = marker
    elif field == "doc_string":
        model.doc_string = marker
    else:
        helper.set_model_props(model, {"key": marker})
    checker.check_model(model)
    wire = model.SerializeToString()
    assert wire.count(marker.encode()) == 1
    replacement = bad_utf8 + b"x" * (len(marker) - len(bad_utf8))
    corrupted = wire.replace(marker.encode(), replacement)

    with pytest.raises(google.protobuf.message.DecodeError):
        onnx.load_model_from_string(corrupted)
    # Pass bytes directly so this exercises the C++ parser, not Python's.
    with pytest.raises(ValueError, match="Unable to parse proto"):
        checker.check_model(corrupted)
    path = tmp_path / "invalid.onnx"
    path.write_bytes(corrupted)
    with pytest.raises(checker.ValidationError, match="Unable to parse proto"):
        checker.check_model(path)


def test_valid_unicode_model_round_trip():
    model = helper.make_model(helper.make_graph([], "圖形\U0001f600", [], []))
    model.doc_string = "說明"
    helper.set_model_props(model, {"作者": "測試"})
    wire = model.SerializeToString()
    assert onnx.load_model_from_string(wire) == model
    checker.check_model(wire)
