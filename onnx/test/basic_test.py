# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import io
import os
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

    def test_save_and_load_model(self) -> None:
        proto = self._simple_model()
        cls = ModelProto
        proto_string = onnx._serialize(proto)

        # Test if input is string
        loaded_proto = onnx.load_model_from_string(proto_string)
        self.assertEqual(proto, loaded_proto)

        # Test if input has a read function
        f = io.BytesIO()
        onnx.save_model(proto_string, f)
        f = io.BytesIO(f.getvalue())
        loaded_proto = onnx.load_model(f, cls)
        self.assertEqual(proto, loaded_proto)

        # Test if input is a file name
        try:
            with tempfile.NamedTemporaryFile(delete=False) as fi:
                onnx.save_model(proto, fi)

            loaded_proto = onnx.load_model(fi.name, cls)
            self.assertEqual(proto, loaded_proto)
        finally:
            os.remove(fi.name)

    def test_save_and_load_tensor(self) -> None:
        proto = self._simple_tensor()
        cls = TensorProto
        proto_string = onnx._serialize(proto)

        # Test if input is string
        loaded_proto = onnx.load_tensor_from_string(proto_string)
        self.assertEqual(proto, loaded_proto)

        # Test if input has a read function
        f = io.BytesIO()
        onnx.save_tensor(loaded_proto, f)
        f = io.BytesIO(f.getvalue())
        loaded_proto = onnx.load_tensor(f, cls)
        self.assertEqual(proto, loaded_proto)

        # Test if input is a file name
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tfile:
                onnx.save_tensor(proto, tfile)

            loaded_proto = onnx.load_tensor(tfile.name, cls)
            self.assertEqual(proto, loaded_proto)
        finally:
            os.remove(tfile.name)

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
