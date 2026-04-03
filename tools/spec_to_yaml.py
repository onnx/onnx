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

# Explicit attribute lists per type, matching the nanobind/pybind11 bindings
# in onnx/cpp2py_export.cc and Python-added properties in onnx/defs/__init__.py.

_OP_SCHEMA_ATTRS = [
    "name",
    "domain",
    "since_version",
    "doc",
    "deprecated",
    "support_level",
    "node_determinism",
    "inputs",
    "outputs",
    "attributes",
    "type_constraints",
    "min_input",
    "max_input",
    "min_output",
    "max_output",
    "has_function",
    "has_context_dependent_function",
    "has_type_and_shape_inference_function",
    "has_data_propagation_function",
    "function_opset_versions",
    "context_dependent_function_opset_versions",
]

_ATTRIBUTE_ATTRS = [
    "name",
    "description",
    "type",
    "default_value",
    "required",
]

_FORMAL_PARAMETER_ATTRS = [
    "name",
    "type_str",
    "types",
    "description",
    "option",
    "is_homogeneous",
    "min_arity",
    "differentiation_category",
]

_TYPE_CONSTRAINT_PARAM_ATTRS = [
    "type_param_str",
    "allowed_type_strs",
    "description",
]

_ATTRS_BY_TYPE: dict[type, list[str]] = {}


def _get_attrs_for(onnx_obj: Any) -> list[str]:
    """Return the explicit attribute list for the given ONNX object type."""
    # Lazily populate the mapping (types aren't available at import time)
    if not _ATTRS_BY_TYPE:
        _ATTRS_BY_TYPE[onnx.defs.OpSchema] = _OP_SCHEMA_ATTRS
        _ATTRS_BY_TYPE[onnx.defs.OpSchema.Attribute] = _ATTRIBUTE_ATTRS
        _ATTRS_BY_TYPE[onnx.defs.OpSchema.FormalParameter] = _FORMAL_PARAMETER_ATTRS
        _ATTRS_BY_TYPE[onnx.defs.OpSchema.TypeConstraintParam] = (
            _TYPE_CONSTRAINT_PARAM_ATTRS
        )
    return _ATTRS_BY_TYPE[type(onnx_obj)]


def dump_onnx_object(
    onnx_obj: onnx.defs.OpSchema
    | onnx.defs.OpSchema.Attribute
    | onnx.defs.OpSchema.FormalParameter
    | onnx.defs.OpSchema.TypeConstraintParam,
) -> dict[str, Any]:
    res = {}
    for attr in _get_attrs_for(onnx_obj):
        value = getattr(onnx_obj, attr)
        if attr == "default_value" and isinstance(
            onnx_obj, onnx.defs.OpSchema.Attribute
        ):
            value = onnx.helper.get_attribute_value(value)
        value = dump_value(value)
        if value is None:
            continue
        res[attr] = value
    return res


def dump_enum(value: enum.Enum) -> str | None:
    if value.name == "Unknown":
        return None
    return value.name


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
            return {k: dump_value(v) for k, v in sorted(value.items())}
        case bool():
            return value
        case float() | int() | str():
            return value
        case set() | frozenset():
            return sorted(dump_value(v) for v in value)  # type: ignore[type-var]
        case Iterable():
            return type(value)(dump_value(v) for v in value)  # type: ignore[call-arg]

    raise RuntimeError(f"Unhandled type {type(value)}")


def main():
    parser = argparse.ArgumentParser(description="Output ONNX spec in YAML format.")
    parser.add_argument("--output", help="Output directory", required=True)
    args = parser.parse_args()

    schemas = sorted(
        onnx.defs.get_all_schemas_with_history(),
        key=lambda s: (s.domain, s.name, s.since_version),
    )
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    latest_versions: dict[tuple[str, str], int] = {}
    for schema in schemas:
        key = (schema.domain, schema.name)
        if key in latest_versions:
            latest_versions[key] = max(latest_versions[key], schema.since_version)
        else:
            latest_versions[key] = schema.since_version

    for schema in schemas:
        schema_dict = dump_value(schema)
        domain = schema.domain or "ai.onnx"
        outdir = pathlib.Path(args.output) / domain
        key = (schema.domain, schema.name)
        if latest_versions[key] != schema.since_version:
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
