import onnx
from onnx.onnx_pb2 import AttributeProto, NodeProto, GraphProto, ModelProto IR_VERSION
import unittest


class TestProtobufExists(unittest.TestCase):

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
