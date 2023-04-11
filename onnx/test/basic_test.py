# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import io
import tempfile
import unittest

import onnx
from onnx import (
    IR_VERSION,
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    helper,
)


class TestBasicFunctions(unittest.TestCase):
    def _simple_model(self) -> ModelProto:
        # Create a ModelProto.
        model = ModelProto()
        model.ir_version = IR_VERSION
        return model

    def _simple_tensor(self) -> TensorProto:
        # Create a TensorProto.
        tensor = helper.make_tensor(
            name="test-tensor",
            data_type=TensorProto.FLOAT,
            dims=(2, 3, 4),
            vals=[x + 0.5 for x in range(24)],
        )
        return tensor

    def test_save_and_load_model_when_input_is_string(self) -> None:
        proto = self._simple_model()
        proto_string = onnx._serialize(proto, "protobuf")

        # Test if input is string
        loaded_proto = onnx.load_model_from_string(proto_string)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_has_read_function(self) -> None:
        # Test if input has a read function
        proto = self._simple_model()
        proto_string = onnx._serialize(proto, "protobuf")
        f = io.BytesIO()
        onnx.save_model(proto_string, f)
        loaded_proto = onnx.load_model(io.BytesIO(f.getvalue()))
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_is_file_name(self) -> None:
        # Test if input is a file name
        proto = self._simple_model()
        with tempfile.NamedTemporaryFile() as f:
            onnx.save_model(proto, f)
            loaded_proto = onnx.load_model(f.name)
            self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_is_string(self) -> None:
        proto = self._simple_tensor()
        proto_string = onnx._serialize(proto, "protobuf")

        # Test if input is string
        loaded_proto = onnx.load_tensor_from_string(proto_string)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_has_read_function(self) -> None:
        # Test if input has a read function
        proto = self._simple_tensor()
        proto_string = onnx._serialize(proto, "protobuf")
        f = io.BytesIO()
        onnx.save_tensor(onnx.load_tensor_from_string(proto_string), f)
        loaded_proto = onnx.load_tensor(io.BytesIO(f.getvalue()))
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_is_file_name(self) -> None:
        # Test if input is a file name
        proto = self._simple_tensor()
        with tempfile.NamedTemporaryFile() as f:
            onnx.save_tensor(proto, f)
            loaded_proto = onnx.load_tensor(f.name)
            self.assertEqual(proto, loaded_proto)

    def test_existence(self) -> None:
        try:
            AttributeProto  # pylint: disable=pointless-statement
            NodeProto  # pylint: disable=pointless-statement
            GraphProto  # pylint: disable=pointless-statement
            ModelProto  # pylint: disable=pointless-statement
        except Exception as e:  # pylint: disable=broad-except
            self.fail(f"Did not find proper onnx protobufs. Error is: {e}")

    def test_version_exists(self) -> None:
        model = ModelProto()
        # When we create it, graph should not have a version string.
        self.assertFalse(model.HasField("ir_version"))
        # We should touch the version so it is annotated with the current
        # ir version of the running ONNX
        model.ir_version = IR_VERSION
        model_string = model.SerializeToString()
        model.ParseFromString(model_string)
        self.assertTrue(model.HasField("ir_version"))
        # Check if the version is correct.
        self.assertEqual(model.ir_version, IR_VERSION)


if __name__ == "__main__":
    unittest.main()
