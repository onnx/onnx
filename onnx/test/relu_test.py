import unittest

from onnx import defs, helper


class TestRelu(unittest.TestCase):

    def test_relu(self):  # type: () -> None
        self.assertTrue(defs.has('Relu'))
        helper.make_node(
            'Relu', ['X'], ['Y'])


if __name__ == '__main__':
    unittest.main()
