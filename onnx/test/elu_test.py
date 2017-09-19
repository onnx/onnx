import unittest

from onnx import defs, helper


class TestRelu(unittest.TestCase):

    def test_elu(self):
        self.assertTrue(defs.has('Elu'))
        schema = defs.get_schema("Elu")
        node_def = helper.make_node(
            'Elu', ['X'], ['Y'], alpha=1.0)
        self.assertTrue(defs.verify(schema, node_def))


if __name__ == '__main__':
    unittest.main()
