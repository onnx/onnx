from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import AttributeProto, NodeProto, GraphProto, ModelProto, IR_VERSION

import io
import onnx
import os
import tempfile
import unittest


class TestProtobufExists(unittest.TestCase):

    def test_load(self):
        # Create a model proto.
        model = ModelProto()
        model.ir_version = IR_VERSION
        model_string = model.SerializeToString()

        # Test if input is string
        loaded_model = onnx.load_from_string(model_string)
        self.assertTrue(model == loaded_model)

        # Test if input has a read function
        f = io.BytesIO(model_string)
        loaded_model = onnx.load(f)
        self.assertTrue(model == loaded_model)

        # Test if input is a file name
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(model_string)
        f.close()
        loaded_model = onnx.load(f.name)
        self.assertTrue(model == loaded_model)
        os.remove(f.name)

    def test_existence(self):
        try:
            AttributeProto
            NodeProto
            GraphProto
            ModelProto
        except Exception as e:
            self.fail(
                'Did not find proper onnx protobufs. Error is: {}'
                .format(e))

    def test_version_exists(self):
        model = ModelProto()
        # When we create it, graph should not have a version string.
        self.assertFalse(model.HasField('ir_version'))
        # We should touch the version so it is annotated with the current
        # ir version of the running ONNX
        model.ir_version = IR_VERSION
        model_string = model.SerializeToString()
        model.ParseFromString(model_string)
        self.assertTrue(model.HasField('ir_version'))
        # Check if the version is correct.
        self.assertEqual(model.ir_version, IR_VERSION)


if __name__ == '__main__':
    unittest.main()
