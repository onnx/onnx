import onnx
from onnx.onnx_pb2 import AttributeProto, NodeProto, GraphProto, IR_VERSION
import unittest


class TestProtobufExists(unittest.TestCase):

    def test_existence(self):
        try:
            AttributeProto
            NodeProto
            GraphProto
        except Exception as e:
            self.fail(
                'Did not find proper onnx protobufs. Error is: {}'
                .format(e))

    def test_version_exists(self):
        graph = GraphProto()
        # When we create it, graph should not have a version string.
        self.assertFalse(graph.HasField('ir_version'))
        # We should touch the version so it is annotated with the current
        # ir version of the running ONNX
        graph.ir_version = IR_VERSION
        graph_string = graph.SerializeToString()
        graph.ParseFromString(graph_string)
        self.assertTrue(graph.HasField('ir_version'))
        # Check if the version is correct.
        self.assertEqual(graph.ir_version, IR_VERSION)


if __name__ == '__main__':
    unittest.main()
