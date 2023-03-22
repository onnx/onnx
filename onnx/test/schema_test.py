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
        schema.add_attribute(
            "TestAttr",
            defs.OpSchema.AttrType.INT,
            description="TestAttr doc",
            required=True,
        )
        self.assertEqual(len(schema.attributes), 1)
        self.assertEqual(schema.attributes["TestAttr"].name, "TestAttr")
        self.assertEqual(schema.attributes["TestAttr"].description, "TestAttr doc")
        self.assertEqual(schema.attributes["TestAttr"].type, defs.OpSchema.AttrType.INT)
        self.assertEqual(schema.attributes["TestAttr"].default_value.i, 0)

    def test_add_attribute_can_specify_default_value(self) -> None:
        # TODO(justinchuby)
        ...

    def test_add_input_adds_one_input(self) -> None:
        schema = defs.OpSchema()
        schema.add_input(0, "TestInput", "T")
        self.assertEqual(len(schema.inputs), 1)
        self.assertEqual(schema.inputs[0].name, "TestInput")
        self.assertEqual(schema.inputs[0].description, "")
        self.assertEqual(schema.inputs[0].typeStr, "T")

    def test_add_input_adds_two_inputs(self) -> None:
        schema = defs.OpSchema()
        schema.add_input(0, "TestInput1", "T", description="TestInput1 doc")
        schema.add_input(1, "TestInput2", "T", description="TestInput2 doc")
        self.assertEqual(len(schema.inputs), 2)
        self.assertEqual(schema.inputs[0].name, "TestInput1")
        self.assertEqual(schema.inputs[0].description, "TestInput1 doc")
        self.assertEqual(schema.inputs[0].typeStr, "T")
        self.assertEqual(schema.inputs[1].name, "TestInput2")
        self.assertEqual(schema.inputs[1].description, "TestInput2 doc")
        self.assertEqual(schema.inputs[1].typeStr, "T")

    def test_add_output_adds_one_output(self) -> None:
        schema = defs.OpSchema()
        schema.add_output(0, "TestOutput", "T")
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(schema.outputs[0].name, "TestOutput")
        self.assertEqual(schema.outputs[0].description, "")
        self.assertEqual(schema.outputs[0].typeStr, "T")

    def test_add_output_adds_two_outputs(self) -> None:
        schema = defs.OpSchema()
        schema.add_output(0, "TestOutput1", "T", description="TestOutput1 doc")
        schema.add_output(1, "TestOutput2", "T", description="TestOutput2 doc")
        self.assertEqual(len(schema.outputs), 2)
        self.assertEqual(schema.outputs[0].name, "TestOutput1")
        self.assertEqual(schema.outputs[0].description, "TestOutput1 doc")
        self.assertEqual(schema.outputs[0].typeStr, "T")
        self.assertEqual(schema.outputs[1].name, "TestOutput2")
        self.assertEqual(schema.outputs[1].description, "TestOutput2 doc")
        self.assertEqual(schema.outputs[1].typeStr, "T")


if __name__ == "__main__":
    unittest.main()
