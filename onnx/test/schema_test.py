# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
import unittest
from typing import Sequence

import parameterized

import onnx
from onnx import defs


class TestSchema(unittest.TestCase):
    def test_get_schema(self) -> None:
        defs.get_schema("Relu")

    def test_typecheck(self) -> None:
        defs.get_schema("Conv")

    def test_attr_default_value(self) -> None:
        v = defs.get_schema("BatchNormalization").attributes["epsilon"].default_value
        self.assertEqual(type(v), onnx.AttributeProto)
        self.assertEqual(v.type, onnx.AttributeProto.FLOAT)

    def test_function_body(self) -> None:
        self.assertEqual(
            type(defs.get_schema("Selu").function_body), onnx.FunctionProto
        )


class TestOpSchema(unittest.TestCase):
    def test_init(self):
        # Test that the constructor creates an OpSchema object
        schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertIsInstance(schema, defs.OpSchema)

    def test_init_with_inputs(self) -> None:
        op_schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            inputs=[defs.OpSchema.FormalParameter("input1", "T")],
            type_constraints=[("T", ["tensor(int64)"], "")],
        )
        self.assertEqual(op_schema.name, "test_op")
        self.assertEqual(op_schema.domain, "test_domain")
        self.assertEqual(op_schema.since_version, 1)
        self.assertEqual(len(op_schema.inputs), 1)
        self.assertEqual(op_schema.inputs[0].name, "input1")
        self.assertEqual(op_schema.inputs[0].type_str, "T")
        self.assertEqual(len(op_schema.type_constraints), 1)
        self.assertEqual(op_schema.type_constraints[0].type_param_str, "T")
        self.assertEqual(
            op_schema.type_constraints[0].allowed_type_strs, ["tensor(int64)"]
        )

    def test_init_creates_multi_input_output_schema(self) -> None:
        op_schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            inputs=[
                defs.OpSchema.FormalParameter("input1", "T"),
                defs.OpSchema.FormalParameter("input2", "T"),
            ],
            outputs=[
                defs.OpSchema.FormalParameter("output1", "T"),
                defs.OpSchema.FormalParameter("output2", "T"),
            ],
            type_constraints=[("T", ["tensor(int64)"], "")],
            attributes=[
                defs.OpSchema.Attribute(
                    "attr1", defs.OpSchema.AttrType.INTS, "attr1 description"
                )
            ],
        )
        self.assertEqual(len(op_schema.inputs), 2)
        self.assertEqual(op_schema.inputs[0].name, "input1")
        self.assertEqual(op_schema.inputs[0].type_str, "T")
        self.assertEqual(op_schema.inputs[1].name, "input2")
        self.assertEqual(op_schema.inputs[1].type_str, "T")
        self.assertEqual(len(op_schema.outputs), 2)
        self.assertEqual(op_schema.outputs[0].name, "output1")
        self.assertEqual(op_schema.outputs[0].type_str, "T")
        self.assertEqual(op_schema.outputs[1].name, "output2")
        self.assertEqual(op_schema.outputs[1].type_str, "T")
        self.assertEqual(len(op_schema.type_constraints), 1)
        self.assertEqual(op_schema.type_constraints[0].type_param_str, "T")
        self.assertEqual(
            op_schema.type_constraints[0].allowed_type_strs, ["tensor(int64)"]
        )
        self.assertEqual(len(op_schema.attributes), 1)
        self.assertEqual(op_schema.attributes["attr1"].name, "attr1")
        self.assertEqual(
            op_schema.attributes["attr1"].type, defs.OpSchema.AttrType.INTS
        )
        self.assertEqual(op_schema.attributes["attr1"].description, "attr1 description")

    def test_init_without_optional_arguments(self) -> None:
        op_schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertEqual(op_schema.name, "test_op")
        self.assertEqual(op_schema.domain, "test_domain")
        self.assertEqual(op_schema.since_version, 1)
        self.assertEqual(len(op_schema.inputs), 0)
        self.assertEqual(len(op_schema.outputs), 0)
        self.assertEqual(len(op_schema.type_constraints), 0)

    def test_name(self):
        # Test that the name parameter is required and is a string
        with self.assertRaises(TypeError):
            defs.OpSchema(domain="test_domain", since_version=1)  # type: ignore
        with self.assertRaises(TypeError):
            defs.OpSchema(123, "test_domain", 1)  # type: ignore

        schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertEqual(schema.name, "test_op")

    def test_domain(self):
        # Test that the domain parameter is required and is a string
        with self.assertRaises(TypeError):
            defs.OpSchema(name="test_op", since_version=1)  # type: ignore
        with self.assertRaises(TypeError):
            defs.OpSchema("test_op", 123, 1)  # type: ignore

        schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertEqual(schema.domain, "test_domain")

    def test_since_version(self):
        # Test that the since_version parameter is required and is an integer
        with self.assertRaises(TypeError):
            defs.OpSchema("test_op", "test_domain")  # type: ignore

        schema = defs.OpSchema("test_op", "test_domain", 1)
        self.assertEqual(schema.since_version, 1)

    def test_doc(self):
        schema = defs.OpSchema("test_op", "test_domain", 1, doc="test_doc")
        self.assertEqual(schema.doc, "test_doc")

    def test_inputs(self):
        # Test that the inputs parameter is optional and is a sequence of FormalParameter tuples
        inputs = [
            defs.OpSchema.FormalParameter(
                name="input1", type_str="T", description="The first input."
            )
        ]
        schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            inputs=inputs,
            type_constraints=[("T", ["tensor(int64)"], "")],
        )

        self.assertEqual(len(schema.inputs), 1)
        self.assertEqual(schema.inputs[0].name, "input1")
        self.assertEqual(schema.inputs[0].type_str, "T")
        self.assertEqual(schema.inputs[0].description, "The first input.")

    def test_outputs(self):
        # Test that the outputs parameter is optional and is a sequence of FormalParameter tuples
        outputs = [
            defs.OpSchema.FormalParameter(
                name="output1", type_str="T", description="The first output."
            )
        ]

        schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            outputs=outputs,
            type_constraints=[("T", ["tensor(int64)"], "")],
        )
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(schema.outputs[0].name, "output1")
        self.assertEqual(schema.outputs[0].type_str, "T")
        self.assertEqual(schema.outputs[0].description, "The first output.")


class TestFormalParameter(unittest.TestCase):
    def test_init(self):
        name = "input1"
        type_str = "tensor(float)"
        description = "The first input."
        param_option = defs.OpSchema.FormalParameterOption.Single
        is_homogeneous = True
        min_arity = 1
        differentiation_category = defs.OpSchema.DifferentiationCategory.Unknown
        formal_parameter = defs.OpSchema.FormalParameter(
            name,
            type_str,
            description,
            param_option=param_option,
            is_homogeneous=is_homogeneous,
            min_arity=min_arity,
            differentiation_category=differentiation_category,
        )

        self.assertEqual(formal_parameter.name, name)
        self.assertEqual(formal_parameter.type_str, type_str)
        self.assertEqual(formal_parameter.description, description)
        self.assertEqual(formal_parameter.option, param_option)
        self.assertEqual(formal_parameter.is_homogeneous, is_homogeneous)
        self.assertEqual(formal_parameter.min_arity, min_arity)
        self.assertEqual(
            formal_parameter.differentiation_category, differentiation_category
        )


class TestTypeConstraintParam(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("single_type", "T", ["tensor(float)"], "Test description"),
            (
                "double_types",
                "T",
                ["tensor(float)", "tensor(int64)"],
                "Test description",
            ),
            ("tuple", "T", ("tensor(float)", "tensor(int64)"), "Test description"),
        ]
    )
    def test_init(
        self,
        _: str,
        type_param_str: str,
        allowed_types: Sequence[str],
        description: str,
    ) -> None:
        type_constraint = defs.OpSchema.TypeConstraintParam(
            type_param_str, allowed_types, description
        )
        self.assertEqual(type_constraint.description, description)
        self.assertEqual(type_constraint.allowed_type_strs, list(allowed_types))
        self.assertEqual(type_constraint.type_param_str, type_param_str)


class TestAttribute(unittest.TestCase):
    def test_init(self):
        name = "test_attr"
        type_ = defs.OpSchema.AttrType.STRINGS
        description = "Test attribute"
        attribute = defs.OpSchema.Attribute(name, type_, description)

        self.assertEqual(attribute.name, name)
        self.assertEqual(attribute.type, type_)
        self.assertEqual(attribute.description, description)

    def test_init_with_default_value(self):
        default_value = (
            defs.get_schema("BatchNormalization").attributes["epsilon"].default_value
        )
        self.assertIsInstance(default_value, onnx.AttributeProto)
        attribute = defs.OpSchema.Attribute("attr1", default_value, "attr1 description")
        self.assertEqual(default_value, attribute.default_value)
        self.assertEqual("attr1", attribute.name)
        self.assertEqual("attr1 description", attribute.description)


if __name__ == "__main__":
    unittest.main()
