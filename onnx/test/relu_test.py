import unittest

from onnx import defs, helper
from onnx.onnx_ml_pb2 import NodeProto


class TestRelu(unittest.TestCase):

    def test_relu(self):
        self.assertTrue(defs.has('Relu'))
        node_def = helper.make_node(
            'Relu', ['X'], ['Y'])

if __name__ == '__main__':
    unittest.main()
