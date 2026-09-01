# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

import onnx
from onnx import TensorProto, defs, helper

MOD_OPSET_13 = 13
MOD_OPSET_28 = 28

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestSchema:
    @staticmethod
    def _tensor_type_proto(elem_type: int) -> onnx.TypeProto:
        type_proto = onnx.TypeProto()
        type_proto.tensor_type.elem_type = elem_type
        return type_proto

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

    def test_mean_variance_normalization_opset28_schema(self) -> None:
        schema = defs.get_schema("MeanVarianceNormalization", 28)
        old_schema = defs.get_schema("MeanVarianceNormalization", 27)

        assert schema.since_version == 28
        assert schema.attributes["epsilon"].default_value.f == pytest.approx(1e-9)
        assert tuple(schema.attributes["axes"].default_value.ints) == (0, 2, 3)
        assert schema.has_context_dependent_function
        assert (
            schema.node_determinism == defs.OpSchema.NodeDeterminism.Deterministic
        )
        assert "epsilon" not in old_schema.attributes

        node = helper.make_node(
            "MeanVarianceNormalization",
            ["X"],
            ["Y"],
            axes=[1, -1],
            epsilon=1e-5,
        )
        function_proto = onnx.FunctionProto()
        function_proto.ParseFromString(
            schema.get_context_dependent_function(
                node.SerializeToString(),
                [self._tensor_type_proto(TensorProto.DOUBLE).SerializeToString()],
            )
        )
        cast_like_outputs = {
            output
            for n in function_proto.node
            if n.op_type == "CastLike"
            for output in n.output
        }
        assert cast_like_outputs == {"Exponent", "Epsilon", "Y"}

    def test_function_body(self) -> None:
        selu_schema = defs.get_schema("Selu")
        assert type(selu_schema.function_body) is onnx.FunctionProto
        assert (
            selu_schema.node_determinism == defs.OpSchema.NodeDeterminism.Deterministic
        )

    @pytest.mark.parametrize("op_version", [23, 24])
    @pytest.mark.parametrize(
        "elem_type",
        [TensorProto.BFLOAT16, TensorProto.FLOAT16, TensorProto.DOUBLE],
    )
    def test_attention_context_dependent_function_with_typed_causal_mask(
        self, op_version: int, elem_type: int
    ) -> None:
        schema = defs.get_schema("Attention", op_version)
        assert schema.has_context_dependent_function
        node = helper.make_node(
            "Attention",
            ["Q", "K", "V", "attn_mask"],
            ["Y"],
            is_causal=1,
            q_num_heads=2,
            kv_num_heads=2,
        )

        input_types = [self._tensor_type_proto(elem_type)] * 4
        function_proto = onnx.FunctionProto()
        function_proto.ParseFromString(
            schema.get_context_dependent_function(
                node.SerializeToString(),
                [input_type.SerializeToString() for input_type in input_types],
            )
        )

        assert any(
            n.op_type == "CastLike"
            and tuple(n.input) == ("MaskTriFloat", "AttnBias")
            and tuple(n.output) == ("MaskTri",)
            for n in function_proto.node
        )
        output_types = onnx.shape_inference.infer_function_output_types(
            function_proto, input_types, list(node.attribute)
        )
        assert output_types[0].tensor_type.elem_type == elem_type

    @pytest.mark.parametrize(
        "elem_type",
        [TensorProto.BFLOAT16, TensorProto.FLOAT16, TensorProto.DOUBLE],
    )
    def test_attention_context_dependent_function_with_typed_padding_mask(
        self, elem_type: int
    ) -> None:
        schema = defs.get_schema("Attention", 24)
        assert schema.has_context_dependent_function
        node = helper.make_node(
            "Attention",
            ["Q", "K", "V", "attn_mask", "", "", "nonpad_kv_seqlen"],
            ["Y"],
            q_num_heads=2,
            kv_num_heads=2,
        )

        input_types = [self._tensor_type_proto(elem_type)] * 4 + [
            self._tensor_type_proto(TensorProto.UNDEFINED),
            self._tensor_type_proto(TensorProto.UNDEFINED),
            self._tensor_type_proto(TensorProto.INT64),
        ]
        function_proto = onnx.FunctionProto()
        function_proto.ParseFromString(
            schema.get_context_dependent_function(
                node.SerializeToString(),
                [input_type.SerializeToString() for input_type in input_types],
            )
        )

        assert any(
            n.op_type == "CastLike"
            and tuple(n.input) == ("PaddingMask4DFloat", "AttnBiasCausalOrNot")
            and tuple(n.output) == ("PaddingMask4D",)
            for n in function_proto.node
        )
        output_types = onnx.shape_inference.infer_function_output_types(
            function_proto, input_types, list(node.attribute)
        )
        assert output_types[0].tensor_type.elem_type == elem_type

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
        range_schema = defs.get_schema("Range")
        assert (
            range_schema.node_determinism == defs.OpSchema.NodeDeterminism.Deterministic
        )
        assert not range_schema.non_deterministic
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

    def test_bitshift_type_constraints(self) -> None:
        def allowed(schema):
            return next(
                set(t.allowed_type_strs)
                for t in schema.type_constraints
                if t.type_param_str == "T"
            )

        unsigned = {
            "tensor(uint8)",
            "tensor(uint16)",
            "tensor(uint32)",
            "tensor(uint64)",
        }
        signed = {"tensor(int8)", "tensor(int16)", "tensor(int32)", "tensor(int64)"}

        bitshift28 = defs.get_schema("BitShift", 28)
        bitshift11 = defs.get_schema("BitShift", 11)
        assert allowed(bitshift28) == unsigned | signed
        assert allowed(bitshift11) == unsigned
        assert "right shift is an arithmetic shift" in bitshift28.doc
        assert "Y is negative" in bitshift28.doc
        assert "effectively decreased" in bitshift11.doc

    def test_mod_opset28_schema(self) -> None:
        mod13 = defs.get_schema("Mod", MOD_OPSET_13)
        mod28 = defs.get_schema("Mod", MOD_OPSET_28)

        assert mod13.since_version == MOD_OPSET_13
        assert mod28.since_version == MOD_OPSET_28
        assert "floating point" not in mod28.doc
        assert "A - floor(A / B) * B" in mod28.doc
        assert "A - trunc(A / B) * B" in mod28.doc

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

    def test_optional_type_constraints(self) -> None:
        def tensor(ts):
            return {f"tensor({t})" for t in ts}

        def seq(ts):
            return {f"seq({t})" for t in ts}

        def optional(ts):
            return {f"optional({t})" for t in ts}

        dtype15 = {
            "float",
            "uint8",
            "int8",
            "uint16",
            "int16",
            "int32",
            "int64",
            "string",
            "bool",
            "float16",
            "double",
            "uint32",
            "uint64",
            "complex64",
            "complex128",
        }
        dtype28 = dtype15 | {
            "bfloat16",
            "float8e4m3fn",
            "float8e4m3fnuz",
            "float8e5m2",
            "float8e5m2fnuz",
            "uint4",
            "int4",
            "float4e2m1",
            "float8e8m0",
            "uint2",
            "int2",
            "float6e2m3",
            "float6e3m2",
        }
        allowed_types = {
            15: tensor(dtype15) | seq(tensor(dtype15)),
            28: tensor(dtype28) | seq(tensor(dtype28)),
        }
        for version, types in allowed_types.items():
            op = defs.get_schema("Optional", version)
            tc = {t.type_param_str: t.allowed_type_strs for t in op.type_constraints}
            assert len(types) == len(tc["V"])
            assert all(t in tc["V"] for t in types)
            assert len(types) == len(tc["O"])
            assert all(t in tc["O"] for t in optional(types))

    def test_optional_docstrings(self) -> None:
        optional15 = defs.get_schema("Optional", 15).doc
        optional28 = defs.get_schema("Optional", 28).doc
        assert optional15
        assert optional15 == optional28

        has_element15 = defs.get_schema("OptionalHasElement", 15).doc
        has_element18 = defs.get_schema("OptionalHasElement", 18).doc
        has_element28 = defs.get_schema("OptionalHasElement", 28).doc
        assert has_element15 != has_element18
        assert has_element18 == has_element28
        assert "tensor or sequence type" not in has_element15
        assert "tensor or sequence type" in has_element18

        get_element15 = defs.get_schema("OptionalGetElement", 15).doc
        get_element18 = defs.get_schema("OptionalGetElement", 18).doc
        get_element28 = defs.get_schema("OptionalGetElement", 28).doc
        assert get_element15 != get_element18
        assert get_element18 == get_element28
        assert "returns the input" not in get_element15
        assert "returns the input" in get_element18

    def test_optional_has_element_type_constraints(self) -> None:
        def tensor(ts):
            return {f"tensor({t})" for t in ts}

        def seq(ts):
            return {f"seq({t})" for t in ts}

        def optional(ts):
            return {f"optional({t})" for t in ts}

        dtype15 = {
            "float",
            "uint8",
            "int8",
            "uint16",
            "int16",
            "int32",
            "int64",
            "string",
            "bool",
            "float16",
            "double",
            "uint32",
            "uint64",
            "complex64",
            "complex128",
        }
        dtype28 = dtype15 | {
            "bfloat16",
            "float8e4m3fn",
            "float8e4m3fnuz",
            "float8e5m2",
            "float8e5m2fnuz",
            "uint4",
            "int4",
            "float4e2m1",
            "float8e8m0",
            "uint2",
            "int2",
            "float6e2m3",
            "float6e3m2",
        }
        allowed_types = {
            15: tensor(dtype15) | seq(tensor(dtype15)),
            18: tensor(dtype15) | seq(tensor(dtype15)),
            28: tensor(dtype28) | seq(tensor(dtype28)),
        }
        for version, types in allowed_types.items():
            op = defs.get_schema("OptionalHasElement", version)
            tc = {
                t.type_param_str: set(t.allowed_type_strs) for t in op.type_constraints
            }
            o_allowed = optional(types)
            if version > min(allowed_types):
                o_allowed = o_allowed | types
            assert len(o_allowed) == len(tc["O"])
            assert all(t in tc["O"] for t in o_allowed)
            assert tc["B"] == {"tensor(bool)"}

    def test_optional_get_element_type_constraints(self) -> None:
        def tensor(ts):
            return {f"tensor({t})" for t in ts}

        def seq(ts):
            return {f"seq({t})" for t in ts}

        def optional(ts):
            return {f"optional({t})" for t in ts}

        dtype15 = {
            "float",
            "uint8",
            "int8",
            "uint16",
            "int16",
            "int32",
            "int64",
            "string",
            "bool",
            "float16",
            "double",
            "uint32",
            "uint64",
            "complex64",
            "complex128",
        }
        dtype28 = dtype15 | {
            "bfloat16",
            "float8e4m3fn",
            "float8e4m3fnuz",
            "float8e5m2",
            "float8e5m2fnuz",
            "uint4",
            "int4",
            "float4e2m1",
            "float8e8m0",
            "uint2",
            "int2",
            "float6e2m3",
            "float6e3m2",
        }
        allowed_types = {
            15: tensor(dtype15) | seq(tensor(dtype15)),
            18: tensor(dtype15) | seq(tensor(dtype15)),
            28: tensor(dtype28) | seq(tensor(dtype28)),
        }
        for version, types in allowed_types.items():
            op = defs.get_schema("OptionalGetElement", version)
            tc = {
                t.type_param_str: set(t.allowed_type_strs) for t in op.type_constraints
            }
            o_allowed = optional(types)
            if version > min(allowed_types):
                o_allowed = o_allowed | types
            assert len(o_allowed) == len(tc["O"])
            assert all(t in tc["O"] for t in o_allowed)
            assert len(types) == len(tc["V"])
            assert all(t in tc["V"] for t in types)

    def test_optional_ops_accept_ir14_sequence_type(self) -> None:
        sequence_input = helper.make_tensor_sequence_value_info(
            "sequence_input", TensorProto.FLOAT6E2M3, [2, 3]
        )
        sequence_output = helper.make_tensor_sequence_value_info(
            "sequence_output", TensorProto.FLOAT6E2M3, [2, 3]
        )
        has_element = helper.make_tensor_value_info("has_element", TensorProto.BOOL, [])
        graph = helper.make_graph(
            [
                helper.make_node("Optional", ["sequence_input"], ["optional_value"]),
                helper.make_node(
                    "OptionalHasElement", ["optional_value"], ["has_element"]
                ),
                helper.make_node(
                    "OptionalGetElement", ["optional_value"], ["sequence_output"]
                ),
            ],
            "optional_ir14_sequence",
            [sequence_input],
            [has_element, sequence_output],
        )
        model = helper.make_model(
            graph,
            ir_version=14,
            opset_imports=[helper.make_opsetid("", 28)],
        )

        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)


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
        expected_parameter_count = 2
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
        assert len(op_schema.inputs) == expected_parameter_count
        assert op_schema.inputs[0].name == "input1"
        assert op_schema.inputs[0].type_str == "T"
        assert op_schema.inputs[1].name == "input2"
        assert op_schema.inputs[1].type_str == "T"
        assert len(op_schema.outputs) == expected_parameter_count
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
    @pytest.mark.parametrize(
        "allowed_types",
        [
            pytest.param(["tensor(float)"], id="list_single"),
            pytest.param(["tensor(float)", "tensor(int64)"], id="list_multiple"),
            pytest.param(("tensor(float)", "tensor(int64)"), id="tuple_multiple"),
        ],
    )
    def test_init(self, allowed_types: Sequence[str]) -> None:
        type_param_str = "T"
        description = "Test description"
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
