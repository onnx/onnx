# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import io
import os
import pathlib
import tempfile
import unittest
from typing import Literal

import parameterized

import onnx
from onnx import serialization


def _simple_model() -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.ir_version = onnx.IR_VERSION
    return model


def _simple_tensor() -> onnx.TensorProto:
    tensor = onnx.helper.make_tensor(
        name="test-tensor",
        data_type=onnx.TensorProto.FLOAT,
        dims=(2, 3, 4),
        vals=[x + 0.5 for x in range(24)],
    )
    return tensor


@parameterized.parameterized_class(
    [
        {"format": "protobuf"},
        {"format": "textproto"},
    ]
)
class TestIO(unittest.TestCase):
    format: Literal["protobuf", "textproto"]

    def test_load_model_when_input_is_bytes(self) -> None:
        proto = _simple_model()
        proto_string = serialization.registry.get(self.format).serialize_proto(proto)
        loaded_proto = onnx.load_model_from_string(proto_string, format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_has_read_function(self) -> None:
        proto = _simple_model()
        # When the proto is a bytes representation provided to `save_model`,
        # it should always be a serialized binary protobuf representation. Aka. format="protobuf"
        # The saved file format is specified by the `format` argument.
        proto_string = serialization.registry.get("protobuf").serialize_proto(proto)
        f = io.BytesIO()
        onnx.save_model(proto_string, f, format=self.format)
        loaded_proto = onnx.load_model(io.BytesIO(f.getvalue()), format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_is_file_name(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.onnx")
            onnx.save_model(proto, model_path, format=self.format)
            loaded_proto = onnx.load_model(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_is_pathlike(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = pathlib.Path(temp_dir, "model.onnx")
            onnx.save_model(proto, model_path, format=self.format)
            loaded_proto = onnx.load_model(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)

    def test_load_tensor_when_input_is_bytes(self) -> None:
        proto = _simple_tensor()
        proto_string = serialization.registry.get(self.format).serialize_proto(proto)
        loaded_proto = onnx.load_tensor_from_string(proto_string, format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_has_read_function(self) -> None:
        # Test if input has a read function
        proto = _simple_tensor()
        f = io.BytesIO()
        onnx.save_tensor(proto, f, format=self.format)
        loaded_proto = onnx.load_tensor(io.BytesIO(f.getvalue()), format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_is_file_name(self) -> None:
        # Test if input is a file name
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.onnx")
            onnx.save_tensor(proto, model_path, format=self.format)
            loaded_proto = onnx.load_tensor(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_is_pathlike(self) -> None:
        # Test if input is a file name
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = pathlib.Path(temp_dir, "model.onnx")
            onnx.save_tensor(proto, model_path, format=self.format)
            loaded_proto = onnx.load_tensor(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)


class TestBasicFunctions(unittest.TestCase):
    def test_protos_exist(self) -> None:
        # The proto classes should exist
        _ = onnx.AttributeProto
        _ = onnx.NodeProto
        _ = onnx.GraphProto
        _ = onnx.ModelProto

    def test_version_exists(self) -> None:
        model = onnx.ModelProto()
        # When we create it, graph should not have a version string.
        self.assertFalse(model.HasField("ir_version"))
        # We should touch the version so it is annotated with the current
        # ir version of the running ONNX
        model.ir_version = onnx.IR_VERSION
        model_string = model.SerializeToString()
        model.ParseFromString(model_string)
        self.assertTrue(model.HasField("ir_version"))
        # Check if the version is correct.
        self.assertEqual(model.ir_version, onnx.IR_VERSION)


if __name__ == "__main__":
    unittest.main()
