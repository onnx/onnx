# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import parameterized
import pytest

import onnx
from onnx import defs

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestSchema:
    def test_get_schema(self) -> None:
        relu_schema = defs.get_schema("Relu")
        assert (
            relu_schema.node_determinism == defs.OpSchema.NodeDeterminism.Deterministic
        )

    def test_typecheck(self) -> None:
        defs.get_schema("Conv")

    def test_attr_default_value(self) -> None:
        v = defs.get_schema("BatchNormalization").attributes["epsilon"].default_value
        assert type(v) is onnx.AttributeProto
        assert v.type == onnx.AttributeProto.FLOAT

    def test_function_body(self) -> None:
        selu_schema = defs.get_schema("Selu")
        assert type(selu_schema.function_body) is onnx.FunctionProto
        assert (
            selu_schema.node_determinism == defs.OpSchema.NodeDeterminism.Deterministic
        )

    def test_node_determinism(self) -> None:
        rand_schema = defs.get_schema("RandomNormalLike")
        assert (
            rand_schema.node_determinism
            == defs.OpSchema.NodeDeterminism.NonDeterministic
        )
        assert rand_schema.non_deterministic
        bn_schema = defs.get_schema("BatchNormalization")
        assert bn_schema.node_determinism == defs.OpSchema.NodeDeterminism.Deterministic
        assert not bn_schema.non_deterministic
        cast_like_schema = defs.get_schema("CastLike")
        assert (
            cast_like_schema.node_determinism
            == defs.OpSchema.NodeDeterminism.Deterministic
        )
        assert not cast_like_schema.non_deterministic
        if_schema = defs.get_schema("If")
        assert (
            if_schema.node_determinism == defs.OpSchema.NodeDeterminism.NonDeterministic
        )
        assert if_schema.non_deterministic

    def test_celu_type_constraints(self) -> None:
        def allowed(schema):
            return next(
                set(t.allowed_type_strs)
                for t in schema.type_constraints
                if t.type_param_str == "T"
            )

        celu28 = defs.get_schema("Celu", 28)
        assert allowed(celu28) == {
            "tensor(bfloat16)",
            "tensor(float16)",
            "tensor(float)",
            "tensor(double)",
        }
        assert celu28.has_function
        assert allowed(defs.get_schema("Celu", 12)) == {"tensor(float)"}

    def test_range_supported_types(self) -> None:
        """Test Range operator supports all expected numeric types."""
        range_schema = defs.get_schema("Range")

        supported_types = set()
        for constraint in range_schema.type_constraints:
            if constraint.type_param_str == "T":
                supported_types.update(constraint.allowed_type_strs)

        expected_types = {
            "tensor(float16)",
            "tensor(bfloat16)",
            "tensor(float)",
            "tensor(double)",
            "tensor(int16)",
            "tensor(int32)",
            "tensor(int64)",
        }

        for expected_type in expected_types:
            assert expected_type in supported_types, (
                f"Range should support {expected_type}"
            )

        # Verify no unexpected types are supported (regression check)
        allowed_type_families = {
            "float16",
            "bfloat16",
            "float",
            "double",
            "int16",
            "int32",
            "int64",
        }

        for supported_type in supported_types:
            if supported_type.startswith("tensor(") and supported_type.endswith(")"):
                base_type = supported_type[7:-1]
                assert base_type in allowed_type_families, (
                    f"Unexpected type {supported_type} supported by Range"
                )

    def test_range_type_consistency(self) -> None:
        """Test Range operator type constraints are consistent."""
        range_schema = defs.get_schema("Range")

        # All inputs should use the same type constraint "T"
        expected_input_names = ["start", "limit", "delta"]
        assert len(range_schema.inputs) == len(expected_input_names)

        for i, expected_name in enumerate(expected_input_names):
            input_param = range_schema.inputs[i]
            assert input_param.name == expected_name
            assert input_param.type_str == "T", (
                f"Input '{expected_name}' should use type constraint 'T'"
            )

        assert len(range_schema.outputs) == 1
        output_param = range_schema.outputs[0]
        assert output_param.name == "output"
        assert output_param.type_str == "T", "Output should use type constraint 'T'"

        type_constraints = [
            c for c in range_schema.type_constraints if c.type_param_str == "T"
        ]
        assert len(type_constraints) == 1, (
            "Range should have exactly one type constraint 'T'"
        )

    def test_range_numeric_types_only(self) -> None:
        """Test Range operator only supports appropriate numeric types."""
        range_schema = defs.get_schema("Range")

        supported_types = set()
        for constraint in range_schema.type_constraints:
            if constraint.type_param_str == "T":
                supported_types.update(constraint.allowed_type_strs)

        unsupported_types = {
            "tensor(bool)",
            "tensor(string)",
            "tensor(uint8)",
            "tensor(uint16)",
            "tensor(uint32)",
            "tensor(uint64)",
            "tensor(int8)",
        }

        for unsupported_type in unsupported_types:
            assert unsupported_type not in supported_types, (
                f"Range should not support {unsupported_type}"
            )

        # All supported types should be appropriate for arithmetic operations
        for supported_type in supported_types:
            assert supported_type.startswith("tensor("), (
                f"All Range types should be tensors, got {supported_type}"
            )

            base_type = supported_type[7:-1]
            assert base_type in [
                "float16",
                "bfloat16",
                "float",
                "double",
                "int16",
                "int32",
                "int64",
            ], f"Range type {base_type} should be a supported numeric type"


class TestOpSchema:
    def test_init(self):
        # Test that the constructor creates an OpSchema object
        schema = defs.OpSchema("test_op", "test_domain", 1)
        assert isinstance(schema, defs.OpSchema)
        assert schema.node_determinism == defs.OpSchema.NodeDeterminism.Deterministic

    def test_init_with_inputs(self) -> None:
        op_schema = defs.OpSchema(
            "test_op",
            "test_domain",
            1,
            inputs=[defs.OpSchema.FormalParameter("input1", "T")],
            type_constraints=[("T", ["tensor(int64)"], "")],
        )
        assert op_schema.name == "test_op"
        assert op_schema.domain == "test_domain"
        assert op_schema.since_version == 1
        assert len(op_schema.inputs) == 1
        assert op_schema.inputs[0].name == "input1"
        assert op_schema.inputs[0].type_str == "T"
        assert len(op_schema.type_constraints) == 1
        assert op_schema.type_constraints[0].type_param_str == "T"
        assert op_schema.type_constraints[0].allowed_type_strs == ["tensor(int64)"]

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
        assert len(op_schema.inputs) == 2
        assert op_schema.inputs[0].name == "input1"
        assert op_schema.inputs[0].type_str == "T"
        assert op_schema.inputs[1].name == "input2"
        assert op_schema.inputs[1].type_str == "T"
        assert len(op_schema.outputs) == 2
        assert op_schema.outputs[0].name == "output1"
        assert op_schema.outputs[0].type_str == "T"
        assert op_schema.outputs[1].name == "output2"
        assert op_schema.outputs[1].type_str == "T"
        assert len(op_schema.type_constraints) == 1
        assert op_schema.type_constraints[0].type_param_str == "T"
        assert op_schema.type_constraints[0].allowed_type_strs == ["tensor(int64)"]
        assert len(op_schema.attributes) == 1
        assert op_schema.attributes["attr1"].name == "attr1"
        assert op_schema.attributes["attr1"].type == defs.OpSchema.AttrType.INTS
        assert op_schema.attributes["attr1"].description == "attr1 description"

    def test_init_without_optional_arguments(self) -> None:
        op_schema = defs.OpSchema("test_op", "test_domain", 1)
        assert op_schema.name == "test_op"
        assert op_schema.domain == "test_domain"
        assert op_schema.since_version == 1
        assert len(op_schema.inputs) == 0
        assert len(op_schema.outputs) == 0
        assert len(op_schema.type_constraints) == 0

    def test_name(self):
        # Test that the name parameter is required and is a string
        with pytest.raises(TypeError):
            defs.OpSchema(domain="test_domain", since_version=1)
        with pytest.raises(TypeError):
            defs.OpSchema(123, "test_domain", 1)

        schema = defs.OpSchema("test_op", "test_domain", 1)
        assert schema.name == "test_op"

    def test_domain(self):
        # Test that the domain parameter is required and is a string
        with pytest.raises(TypeError):
            defs.OpSchema(name="test_op", since_version=1)
        with pytest.raises(TypeError):
            defs.OpSchema("test_op", 123, 1)

        schema = defs.OpSchema("test_op", "test_domain", 1)
        assert schema.domain == "test_domain"

    def test_since_version(self):
        # Test that the since_version parameter is required and is an integer
        with pytest.raises(TypeError):
            defs.OpSchema("test_op", "test_domain")

        schema = defs.OpSchema("test_op", "test_domain", 1)
        assert schema.since_version == 1

    def test_doc(self):
        schema = defs.OpSchema("test_op", "test_domain", 1, doc="test_doc")
        assert schema.doc == "test_doc"

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

        assert len(schema.inputs) == 1
        assert schema.inputs[0].name == "input1"
        assert schema.inputs[0].type_str == "T"
        assert schema.inputs[0].description == "The first input."

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
        assert len(schema.outputs) == 1
        assert schema.outputs[0].name == "output1"
        assert schema.outputs[0].type_str == "T"
        assert schema.outputs[0].description == "The first output."


class TestFormalParameter:
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

        assert formal_parameter.name == name
        assert formal_parameter.type_str == type_str
        assert isinstance(formal_parameter.types, set)
        assert formal_parameter.description == description
        assert formal_parameter.option == param_option
        assert formal_parameter.is_homogeneous == is_homogeneous
        assert formal_parameter.min_arity == min_arity
        assert formal_parameter.differentiation_category == differentiation_category


class TestTypeConstraintParam:
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
        assert type_constraint.description == description
        assert type_constraint.allowed_type_strs == list(allowed_types)
        assert type_constraint.type_param_str == type_param_str


class TestAttribute:
    def test_init(self):
        name = "test_attr"
        type_ = defs.OpSchema.AttrType.STRINGS
        description = "Test attribute"
        attribute = defs.OpSchema.Attribute(name, type_, description)

        assert attribute.name == name
        assert attribute.type == type_
        assert attribute.description == description

    def test_init_with_default_value(self):
        default_value = (
            defs.get_schema("BatchNormalization").attributes["epsilon"].default_value
        )
        assert isinstance(default_value, onnx.AttributeProto)
        attribute = defs.OpSchema.Attribute("attr1", default_value, "attr1 description")
        assert default_value == attribute.default_value
        assert attribute.name == "attr1"
        assert attribute.description == "attr1 description"


@pytest.mark.parametrize(
    ("op_type", "op_version", "op_domain", "trap_op_version"),
    [
        # register to exist domain
        ("CustomOp", 5, "", [1, 2, 6, 7]),
        # register to new domain
        ("CustomOp", 5, "test", [1, 2, 6, 7]),
    ],
)
class TestOpSchemaRegister:
    op_type: str
    op_version: int
    op_domain: str
    # register some fake schema to check behavior
    trap_op_version: list[int]

    @pytest.fixture(autouse=True)
    def _register_schema(self, op_type, op_version, op_domain, trap_op_version):
        self.op_type = op_type
        self.op_version = op_version
        self.op_domain = op_domain
        self.trap_op_version = trap_op_version
        # Ensure the schema is unregistered
        assert not onnx.defs.has(self.op_type, self.op_domain)
        yield
        # Clean up the registered schema
        for version in [*self.trap_op_version, self.op_version]:
            with contextlib.suppress(onnx.defs.SchemaError):
                onnx.defs.deregister_schema(self.op_type, version, self.op_domain)

    def test_register_multi_schema(self):
        for version in [*self.trap_op_version, self.op_version]:
            op_schema = defs.OpSchema(
                self.op_type,
                self.op_domain,
                version,
            )
            onnx.defs.register_schema(op_schema)
            assert onnx.defs.has(self.op_type, version, self.op_domain)
        for version in [*self.trap_op_version, self.op_version]:
            # Also make sure the `op_schema` is accessible after register
            registered_op = onnx.defs.get_schema(
                op_schema.name, version, op_schema.domain
            )
            op_schema = defs.OpSchema(
                self.op_type,
                self.op_domain,
                version,
            )
            assert str(registered_op) == str(op_schema)

    def test_using_the_specified_version_in_onnx_check(self):
        input = f"""
            <
                ir_version: 7,
                opset_import: [
                    "{self.op_domain}" : {self.op_version}
                ]
            >
            agraph (float[N, 128] X, int32 Y) => (float[N] Z)
            {{
                Z = {self.op_domain}.{self.op_type}<attr1=[1,2]>(X, Y)
            }}
           """
        model = onnx.parser.parse_model(input)
        op_schema = defs.OpSchema(
            self.op_type,
            self.op_domain,
            self.op_version,
            inputs=[
                defs.OpSchema.FormalParameter("input1", "T"),
                defs.OpSchema.FormalParameter("input2", "int32"),
            ],
            outputs=[
                defs.OpSchema.FormalParameter("output1", "T"),
            ],
            type_constraints=[("T", ["tensor(float)"], "")],
            attributes=[
                defs.OpSchema.Attribute(
                    "attr1", defs.OpSchema.AttrType.INTS, "attr1 description"
                )
            ],
        )
        with pytest.raises(onnx.checker.ValidationError):
            onnx.checker.check_model(model, check_custom_domain=True)
        onnx.defs.register_schema(op_schema)
        # The fake schema will raise check exception if selected in checker
        for version in self.trap_op_version:
            onnx.defs.register_schema(
                defs.OpSchema(
                    self.op_type,
                    self.op_domain,
                    version,
                    outputs=[
                        defs.OpSchema.FormalParameter("output1", "int32"),
                    ],
                )
            )
        onnx.checker.check_model(model, check_custom_domain=True)

    def test_register_schema_raises_error_when_registering_a_schema_twice(self):
        op_schema = defs.OpSchema(
            self.op_type,
            self.op_domain,
            self.op_version,
        )
        onnx.defs.register_schema(op_schema)
        with pytest.raises(onnx.defs.SchemaError):
            onnx.defs.register_schema(op_schema)

    def test_deregister_the_specified_schema(self):
        for version in [*self.trap_op_version, self.op_version]:
            op_schema = defs.OpSchema(
                self.op_type,
                self.op_domain,
                version,
            )
            onnx.defs.register_schema(op_schema)
            assert onnx.defs.has(op_schema.name, version, op_schema.domain)
        onnx.defs.deregister_schema(op_schema.name, self.op_version, op_schema.domain)
        for version in self.trap_op_version:
            assert onnx.defs.has(op_schema.name, version, op_schema.domain)
        # Maybe has lesser op version in trap list
        if onnx.defs.has(op_schema.name, self.op_version, op_schema.domain):
            schema = onnx.defs.get_schema(
                op_schema.name, self.op_version, op_schema.domain
            )
            assert schema.since_version < self.op_version

    def test_deregister_schema_raises_error_when_opschema_does_not_exist(self):
        with pytest.raises(onnx.defs.SchemaError):
            onnx.defs.deregister_schema(self.op_type, self.op_version, self.op_domain)

    def test_legacy_schema_accessible_after_deregister(self):
        op_schema = defs.OpSchema(
            self.op_type,
            self.op_domain,
            self.op_version,
        )
        onnx.defs.register_schema(op_schema)
        schema_a = onnx.defs.get_schema(
            op_schema.name, op_schema.since_version, op_schema.domain
        )
        schema_b = onnx.defs.get_schema(op_schema.name, op_schema.domain)

        def filter_schema(schemas):
            return [op for op in schemas if op.name == op_schema.name]

        schema_c = filter_schema(onnx.defs.get_all_schemas())
        schema_d = filter_schema(onnx.defs.get_all_schemas_with_history())
        assert len(schema_c) == 1
        assert len(schema_d) == 1
        # Avoid memory residue and access storage as much as possible
        assert str(schema_a) == str(op_schema)
        assert str(schema_b) == str(op_schema)
        assert str(schema_c[0]) == str(op_schema)
        assert str(schema_d[0]) == str(op_schema)
