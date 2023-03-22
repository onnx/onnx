# SPDX-License-Identifier: Apache-2.0
import unittest

from onnx import AttributeProto, FunctionProto, defs


class TestSchema(unittest.TestCase):
    def test_get_schema(self) -> None:
        defs.get_schema("Relu")

    def test_typecheck(self) -> None:
        defs.get_schema("Conv")

    def test_attr_default_value(self) -> None:
        v = defs.get_schema("BatchNormalization").attributes["epsilon"].default_value
        self.assertEqual(type(v), AttributeProto)
        self.assertEqual(v.type, AttributeProto.FLOAT)

    def test_function_body(self) -> None:
        self.assertEqual(type(defs.get_schema("Selu").function_body), FunctionProto)


class TestOpSchema(unittest.TestCase):
    def test_init_successful(self) -> None:
        defs.OpSchema()

    def test_name_can_be_set(self) -> None:
        schema = defs.OpSchema()
        schema.name = "TestOp"
        self.assertEqual(schema.name, "TestOp")

    def test_doc_can_be_set(self) -> None:
        schema = defs.OpSchema()
        schema.doc = "Test doc"
        self.assertEqual(schema.doc, "Test doc")

    def test_domain_can_be_set(self) -> None:
        schema = defs.OpSchema()
        schema.domain = "TestDomain"
        self.assertEqual(schema.domain, "TestDomain")

    def test_add_attribute_adds_one_attribute(self) -> None:
        schema = defs.OpSchema()
        schema.add_attribute("TestAttr", "TestAttr doc", defs.OpSchema.AttrType.INT, required=True)
        self.assertEqual(len(schema.attributes), 1)
        self.assertEqual(schema.attributes["TestAttr"].name, "TestAttr")
        self.assertEqual(schema.attributes["TestAttr"].description, "TestAttr doc")
        self.assertEqual(schema.attributes["TestAttr"].type, defs.OpSchema.AttrType.INT)
        self.assertEqual(schema.attributes["TestAttr"].default_value.i, 0)

    def test_add_attribute_can_specify_default_value(self) -> None:
        # TODO(justinchuby)
        ...


if __name__ == "__main__":
    unittest.main()
