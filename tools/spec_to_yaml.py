# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Output ONNX spec in YAML format.

Usage:

    python spec_to_yaml.py --output onnx-spec/defs
"""

from __future__ import annotations

import argparse
import enum
import pathlib
from collections.abc import Iterable
from typing import Any

from ruamel.yaml import YAML

import onnx


def dump_onnx_object(
    onnx_obj: onnx.defs.OpSchema
    | onnx.defs.OpSchema.Attribute
    | onnx.defs.OpSchema.FormalParameter
    | onnx.defs.OpSchema.TypeConstraintParam,
) -> dict[str, Any]:
    res = {}
    for attr in dir(onnx_obj):
        if attr.startswith("_"):
            continue
        value = getattr(onnx_obj, attr)
        if isinstance(value, enum.EnumType) or "nanobind" in str(type(value)):
            continue
        if attr == "default_value" and isinstance(
            onnx_obj, onnx.defs.OpSchema.Attribute
        ):
            value = onnx.helper.get_attribute_value(value)
        value = dump_value(value)
        if not value:
            continue
        res[attr] = value
    return res


def dump_enum(value: enum.Enum) -> str | None:
    for member in type(value):
        if member == value:
            if member.name == "Unknown":
                return None
            return member.name
    raise RuntimeError(f"Unhandled type {type(value)}")


def dump_value(value: Any):  # noqa: PLR0911
    match value:
        case None:
            return None
        case (
            onnx.defs.OpSchema()
            | onnx.defs.OpSchema.Attribute()
            | onnx.defs.OpSchema.FormalParameter()
            | onnx.defs.OpSchema.TypeConstraintParam()
        ):
            return dump_onnx_object(value)
        case onnx.FunctionProto():
            return onnx.printer.to_text(value)
        case enum.Enum():
            return dump_enum(value)
        case dict():
            return {k: dump_value(v) for k, v in value.items()}
        case float() | int() | str():
            return value
        case Iterable():
            return type(value)(dump_value(v) for v in value)  # type: ignore

    raise RuntimeError(f"Unhandled type {type(value)}")


def main():
    parser = argparse.ArgumentParser(description="Output ONNX spec in YAML format.")
    parser.add_argument("--output", help="Output directory", required=True)
    args = parser.parse_args()

    schemas = onnx.defs.get_all_schemas_with_history()
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    latest_versions: dict = {}
    for schema in schemas:
        if schema.name in latest_versions:
            latest_versions[schema.name] = max(
                latest_versions[schema.name], schema.since_version
            )
        else:
            latest_versions[schema.name] = schema.since_version
    for schema in schemas:
        schema_dict = dump_value(schema)
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
            yaml.dump(schema_dict, f)


if __name__ == "__main__":
    main()
