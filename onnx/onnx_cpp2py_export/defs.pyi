"""Submodule containing all the ONNX schema definitions."""

from collections.abc import Sequence
from typing import overload

from onnx import AttributeProto, FunctionProto

class SchemaError(Exception): ...

class OpSchema:
    def __init__(
        self,
        name: str,
        domain: str,
        since_version: int,
        doc: str = "",
        *,
        inputs: Sequence[OpSchema.FormalParameter] = (),
        outputs: Sequence[OpSchema.FormalParameter] = (),
        type_constraints: Sequence[tuple[str, Sequence[str], str]] = (),
        attributes: Sequence[OpSchema.Attribute] = (),
    ) -> None: ...
    @property
    def file(self) -> str: ...
    @property
    def line(self) -> int: ...
    @property
    def support_level(self) -> SupportType: ...
    @property
    def doc(self) -> str | None: ...
    @property
    def since_version(self) -> int: ...
    @property
    def deprecated(self) -> bool: ...
    @property
    def domain(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def min_input(self) -> int: ...
    @property
    def max_input(self) -> int: ...
    @property
    def min_output(self) -> int: ...
    @property
    def max_output(self) -> int: ...
    @property
    def attributes(self) -> dict[str, Attribute]: ...
    @property
    def inputs(self) -> Sequence[FormalParameter]: ...
    @property
    def outputs(self) -> Sequence[FormalParameter]: ...
    @property
    def type_constraints(self) -> Sequence[TypeConstraintParam]: ...
    @property
    def has_type_and_shape_inference_function(self) -> bool: ...
    @property
    def has_data_propagation_function(self) -> bool: ...
    @staticmethod
    def is_infinite(v: int) -> bool: ...
    def consumed(self, schema: OpSchema, i: int) -> tuple[UseType, int]: ...
    def _infer_node_outputs(
        self,
        node_proto: bytes,
        value_types: dict[str, bytes],
        input_data: dict[str, bytes],
        input_sparse_data: dict[str, bytes],
    ) -> dict[str, bytes]: ...
    @property
    def function_body(self) -> FunctionProto: ...

    class TypeConstraintParam:
        def __init__(
            self,
            type_param_str: str,
            allowed_type_strs: Sequence[str],
            description: str = "",
        ) -> None:
            """Type constraint parameter.

            Args:
                type_param_str: Type parameter string, for example, "T", "T1", etc.
                allowed_type_strs: Allowed type strings for this type parameter. E.g. ["tensor(float)"].
                description: Type parameter description.
            """
        @property
        def type_param_str(self) -> str: ...
        @property
        def description(self) -> str: ...
        @property
        def allowed_type_strs(self) -> Sequence[str]: ...

    class FormalParameterOption:
        Single: OpSchema.FormalParameterOption = ...
        Optional: OpSchema.FormalParameterOption = ...
        Variadic: OpSchema.FormalParameterOption = ...

    class DifferentiationCategory:
        Unknown: OpSchema.DifferentiationCategory = ...
        Differentiable: OpSchema.DifferentiationCategory = ...
        NonDifferentiable: OpSchema.DifferentiationCategory = ...

    class FormalParameter:
        def __init__(
            self,
            name: str,
            type_str: str,
            description: str = "",
            *,
            param_option: OpSchema.FormalParameterOption = OpSchema.FormalParameterOption.Single,  # noqa: F821
            is_homogeneous: bool = True,
            min_arity: int = 1,
            differentiation_category: OpSchema.DifferentiationCategory = OpSchema.DifferentiationCategory.Unknown,  # noqa: F821
        ) -> None: ...
        @property
        def name(self) -> str: ...
        @property
        def types(self) -> set[str]: ...
        @property
        def type_str(self) -> str: ...
        @property
        def description(self) -> str: ...
        @property
        def option(self) -> OpSchema.FormalParameterOption: ...
        @property
        def is_homogeneous(self) -> bool: ...
        @property
        def min_arity(self) -> int: ...
        @property
        def differentiation_category(self) -> OpSchema.DifferentiationCategory: ...

    class AttrType:
        FLOAT: OpSchema.AttrType = ...
        INT: OpSchema.AttrType = ...
        STRING: OpSchema.AttrType = ...
        TENSOR: OpSchema.AttrType = ...
        GRAPH: OpSchema.AttrType = ...
        SPARSE_TENSOR: OpSchema.AttrType = ...
        TYPE_PROTO: OpSchema.AttrType = ...
        FLOATS: OpSchema.AttrType = ...
        INTS: OpSchema.AttrType = ...
        STRINGS: OpSchema.AttrType = ...
        TENSORS: OpSchema.AttrType = ...
        GRAPHS: OpSchema.AttrType = ...
        SPARSE_TENSORS: OpSchema.AttrType = ...
        TYPE_PROTOS: OpSchema.AttrType = ...

    class Attribute:
        @overload
        def __init__(
            self,
            name: str,
            type: OpSchema.AttrType,
            description: str = "",
            *,
            required: bool = True,
        ) -> None: ...
        @overload
        def __init__(
            self,
            name: str,
            default_value: AttributeProto,
            description: str = "",
        ) -> None: ...
        @property
        def name(self) -> str: ...
        @property
        def description(self) -> str: ...
        @property
        def type(self) -> OpSchema.AttrType: ...
        @property
        def default_value(self) -> AttributeProto: ...
        @property
        def required(self) -> bool: ...

    class SupportType(int):
        COMMON: OpSchema.SupportType = ...
        EXPERIMENTAL: OpSchema.SupportType = ...

    class UseType:
        DEFAULT: OpSchema.UseType = ...
        CONSUME_ALLOWED: OpSchema.UseType = ...
        CONSUME_ENFORCED: OpSchema.UseType = ...

@overload
def has_schema(op_type: str, domain: str = "") -> bool: ...
@overload
def has_schema(op_type: str, max_inclusive_version: int, domain: str = "") -> bool: ...
def schema_version_map() -> dict[str, tuple[int, int]]: ...
@overload
def get_schema(
    op_type: str, max_inclusive_version: int, domain: str = ""
) -> OpSchema: ...
@overload
def get_schema(op_type: str, domain: str = "") -> OpSchema: ...
def get_all_schemas() -> Sequence[OpSchema]: ...
def get_all_schemas_with_history() -> Sequence[OpSchema]: ...
def set_domain_to_version(
    domain: str, min_version: int, max_version: int, last_release_version: int = -1
) -> None: ...
def register_schema(schema: OpSchema) -> None: ...
def deregister_schema(op_type: str, version: int, domain: str) -> None: ...
