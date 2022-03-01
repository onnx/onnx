# SPDX-License-Identifier: Apache-2.0
import unittest

from onnx import defs, AttributeProto


class TestSchema(unittest.TestCase):

    def test_get_schema(self) -> None:
        defs.get_schema("Relu")

    def test_typecheck(self) -> None:
        defs.get_schema("Conv")

    def test_attr_default_value(self) -> None:
        v = defs.get_schema(
            "BatchNormalization").attributes['epsilon'].default_value
        self.assertEqual(type(v), AttributeProto)
        self.assertEqual(v.type, AttributeProto.FLOAT)


if __name__ == '__main__':
    unittest.main()
