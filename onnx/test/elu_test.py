import unittest

from onnx import defs, checker, helper


class TestRelu(unittest.TestCase):

    def test_elu(self):  # type: () -> None
        self.assertTrue(defs.has('Elu'))
        node_def = helper.make_node(
            'Elu', ['X'], ['Y'], alpha=1.0)
        checker.check_node(node_def)


if __name__ == '__main__':
    unittest.main()
