import unittest

from onnx import defs, onnx_ml_pb2, helper


class TestSchema(unittest.TestCase):

    def test_verify(self):
        schema = defs.get_schema("Relu")
        node_def = helper.make_node(
            "Relu", ["X"], ["Y"], name="test")
        self.assertTrue(defs.verify(schema, node_def))

    def test_typecheck(self):
        schema = defs.get_schema("Conv")

        # kernels should be [int]
        node_def = helper.make_node(
            op_type="Conv",
            inputs=["X", "w"],
            outputs=["Y"],
            name="test",
            kernel_shape=["a"])
        self.assertFalse(defs.verify(schema, node_def))

        node_def = helper.make_node(
            op_type="Conv",
            inputs=["X", "w"],
            outputs=["Y"],
            name="test",
            kernel_shape=[1, 2])
        self.assertTrue(defs.verify(schema, node_def))

if __name__ == '__main__':
    unittest.main()
