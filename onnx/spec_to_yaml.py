"""Output ONNX spec in YAML format.

Usage:

    python spec_to_yaml.py --output onnx-spec/defs
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import TYPE_CHECKING

from defs.gen_doc import generate_formal_parameter_tags
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import FoldedScalarString, LiteralScalarString
from typing_extensions import Self

# import textwrap
import onnx

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclasses.dataclass
class Attribute:
    name: str
    description: str
    type: str
    required: bool
    default_value: (
        str | int | float | Iterable[str] | Iterable[int] | Iterable[float] | None
    ) = None

    def as_dict(self):
        d = dataclasses.asdict(self)
        if d["default_value"] is None:
            d.pop("default_value")
        return d


@dataclasses.dataclass
class FormalParameter:
    name: str
    type_str: str
    description: str
    min_arity: int
    tags: list[str]


@dataclasses.dataclass
class TypeConstraintParam:
    type_param_str: str
    description: str
    allowed_type_strs: list[str]


@dataclasses.dataclass
class OpSchema:
    domain: str
    name: str
    support_level: str
    since_version: int
    min_input: int
    max_input: int
    min_output: int
    max_output: int
    doc: str
    attributes: list[Attribute]
    inputs: list[FormalParameter]
    outputs: list[FormalParameter]
    type_constraints: list[TypeConstraintParam]
    function: str | None = None
    deprecated: bool = False

    def as_dict(self):
        d = dataclasses.asdict(self)
        if self.function is None:
            d.pop("function")
        return d

    @classmethod
    def from_onnx_opschema(cls, schema: onnx.defs.OpSchema) -> Self:
        # field_names = {
        #     f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")
        # }
        # scheme_attrs = {attr for attr in dir(schema) if not attr.startswith("_")}
        # assert sorted(field_names) == sorted(scheme_attrs), field_names.difference(
        #     scheme_attrs
        # )
        return cls(
            support_level="COMMON"
            if schema.support_level == onnx.defs.OpSchema.SupportType.COMMON
            else "EXPERIMENTAL",
            doc=LiteralScalarString(_process_documentation(schema.doc)),
            since_version=schema.since_version,
            deprecated=schema.deprecated,
            domain=schema.domain,
            name=schema.name,
            min_input=schema.min_input,
            max_input=schema.max_input,
            min_output=schema.min_output,
            max_output=schema.max_output,
            attributes=[
                Attribute(
                    name=attr.name,
                    description=FoldedScalarString(
                        _process_documentation(attr.description)
                    ),
                    type=str(attr.type).split(".")[-1],
                    required=attr.required,
                    default_value=_get_attribute_default_value(attr),
                )
                for attr in sorted(schema.attributes.values(), key=lambda a: a.name)
            ],
            inputs=[
                FormalParameter(
                    name=input_.name,
                    type_str=input_.type_str,
                    description=input_.description,
                    min_arity=input_.min_arity,
                    tags=generate_formal_parameter_tags(input_),
                )
                for input_ in schema.inputs
            ],
            outputs=[
                FormalParameter(
                    name=output.name,
                    type_str=output.type_str,
                    description=output.description,
                    min_arity=output.min_arity,
                    tags=generate_formal_parameter_tags(output),
                )
                for output in schema.outputs
            ],
            type_constraints=[
                TypeConstraintParam(
                    type_param_str=type_constraint.type_param_str,
                    description=type_constraint.description,
                    allowed_type_strs=list(type_constraint.allowed_type_strs),
                )
                for type_constraint in schema.type_constraints
            ],
            function=LiteralScalarString(onnx.printer.to_text(schema.function_body))
            if schema.has_function
            else None,
        )



def _get_attribute_default_value(attr: onnx.defs.OpSchema.Attribute):
    value = onnx.helper.get_attribute_value(attr.default_value)
    if value is None:
        return None
    if attr.type == onnx.AttributeProto.STRING:
        value = value.decode("utf-8")
    elif attr.type == onnx.AttributeProto.STRINGS:
        value = [v.decode("utf-8") for v in value]
    elif attr.type == onnx.AttributeProto.GRAPH:
        value = onnx.printer.to_text(value)
    elif attr.type == onnx.AttributeProto.GRAPHS:
        value = [onnx.printer.to_text(v) for v in value]
    elif attr.type == onnx.AttributeProto.TENSOR:
        value = onnx.numpy_helper.to_array(value)
    elif attr.type == onnx.AttributeProto.TENSORS:
        value = [onnx.numpy_helper.to_array(v) for v in value]
    elif attr.type in (
        onnx.AttributeProto.FLOAT,
        onnx.AttributeProto.INT,
        onnx.AttributeProto.INTS,
    ):
        pass
    else:
        raise RuntimeError(f"Unhandled type {attr.type}")
    return value


def _process_documentation(doc: str | None) -> str:
    return doc or ""
    # # Lifted from ONNX's docsgen:
    # # https://github.com/onnx/onnx/blob/3fd41d249bb8006935aa0031a332dd945e61b7e5/docs/docsgen/source/onnx_sphinx.py#L414
    # if not doc:
    #     return ""
    # doc = textwrap.dedent(doc)
    # rep = {
    #     "<dl>": "",
    #     "</dl>": "",
    #     "<dt>": "* ",
    #     "<dd>": "  ",
    #     "</dt>": "",
    #     "</dd>": "",
    #     "<tt>": "`",
    #     "</tt>": "`",
    #     "<br>": "\n",
    # }
    # for k, v in rep.items():
    #     doc = doc.replace(k, v)
    # doc = doc.strip()
    # return doc


def main():
    parser = argparse.ArgumentParser(description="Output ONNX spec in YAML format.")
    parser.add_argument("--output", help="Output directory", required=True)
    args = parser.parse_args()

    schemas = onnx.defs.get_all_schemas_with_history()
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    latest_versions = {}
    for schema in schemas:
        if schema.name in latest_versions:
            latest_versions[schema.name] = max(
                latest_versions[schema.name], schema.since_version
            )
        else:
            latest_versions[schema.name] = schema.since_version
    for schema in schemas:
        dataclass_schema = OpSchema.from_onnx_opschema(schema)
        domain = schema.domain or "ai.onnx"
        outdir = pathlib.Path(args.output) / domain
        if latest_versions[schema.name] != schema.since_version:
            outdir = outdir / "old"
        else:
            outdir = outdir / "latest"
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{schema.name}-{schema.since_version}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            print(f"Writing {path}")
            yaml.dump(dataclass_schema.as_dict(), f)


if __name__ == "__main__":
    main()
