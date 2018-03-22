import unittest

from onnx import defs


class TestSchema(unittest.TestCase):

    def test_get_schema(self):
        defs.get_schema("Relu")

    def test_typecheck(self):
        defs.get_schema("Conv")


if __name__ == '__main__':
    unittest.main()
