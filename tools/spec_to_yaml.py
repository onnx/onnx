# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Output ONNX spec in YAML format.

One YAML file per (domain, op, since_version) under
``<output>/<domain>/{latest,old}/<Op>-<version>.yaml``. Maintainer-only.

Policy: emit only what the C++ author wrote. Dropped: derived fields
(min/max_input/output), framework-state booleans (``has_*_function``;
use ``*_opset_versions`` instead), empty containers, C++ defaults
(``deprecated=False``, ``support_level=COMMON``, ``domain=''``,
``node_determinism=Deterministic``, ``FormalParameter`` ``option=Single`` /
``is_homogeneous=True`` / ``min_arity=1``, ``Attribute required=False``),
per-param ``types`` when derivable from ``type_str`` + ``type_constraints``,
and ``Attribute.name`` (duplicates the ``attributes``-map key).
``Deterministic`` is treated as a default because ``GetNodeDeterminism``
infers it for non-control-flow ops, indistinguishably from author-set.

Usage: ``python tools/spec_to_yaml.py --output <output_dir>``
"""

from __future__ import annotations

import argparse
import enum
import pathlib
from collections.abc import Iterable
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

import onnx

# Explicit attr lists per type; mirror cpp2py_export.cc + defs/__init__.py.

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
    "function_opset_versions",
    "context_dependent_function_opset_versions",
]

# `name` omitted: duplicates the attributes-dict key.
_ATTRIBUTE_ATTRS = [
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

# Per-type C++ defaults; fields at their default are omitted (see module docstring).
_DEFAULTS_BY_TYPE_NAME: dict[str, dict[str, Any]] = {
    "OpSchema": {
        "domain": "",
        "deprecated": False,
        "support_level": "COMMON",
        "node_determinism": "Deterministic",
    },
    "FormalParameter": {
        "option": "Single",
        "is_homogeneous": True,
        "min_arity": 1,
    },
    "Attribute": {
        "required": False,
    },
}


def _defaults_for(onnx_obj: Any) -> dict[str, Any]:
    return _DEFAULTS_BY_TYPE_NAME.get(type(onnx_obj).__name__, {})


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
    defaults = _defaults_for(onnx_obj)
    for attr in _get_attrs_for(onnx_obj):
        value = getattr(onnx_obj, attr)
        if attr == "default_value" and isinstance(
            onnx_obj, onnx.defs.OpSchema.Attribute
        ):
            value = onnx.helper.get_attribute_value(value)
        value = dump_value(value)
        if value is None:
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        if attr in defaults and value == defaults[attr]:
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
        case bytes():
            # STRING defaults arrive as raw protobuf bytes; decode for readable YAML.
            return dump_value(value.decode("utf-8"))
        case str():
            # Multi-line strings -> block scalar so doc diffs stay readable.
            return LiteralScalarString(value) if "\n" in value else value
        case float() | int():
            return value
        case set() | frozenset():
            return sorted(dump_value(v) for v in value)  # type: ignore[type-var]
        case Iterable():
            return type(value)(dump_value(v) for v in value)  # type: ignore[call-arg]

    raise RuntimeError(f"Unhandled type {type(value)}")


def _strip_derivable_param_types(schema_dict: dict[str, Any]) -> None:
    """Drop per-param ``types`` when derivable: equals the named TC's sorted
    ``allowed_type_strs``, or equals ``[type_str]`` for a concrete type.
    Mutates ``schema_dict`` in place.
    """
    tc_index = {
        t["type_param_str"]: sorted(t["allowed_type_strs"])
        for t in schema_dict.get("type_constraints", [])
    }
    for section in ("inputs", "outputs"):
        for param in schema_dict.get(section, []):
            type_str = param.get("type_str")
            types = param.get("types")
            if types is None or type_str is None:
                continue
            if type_str in tc_index:
                if sorted(types) == tc_index[type_str]:
                    param.pop("types")
            elif list(types) == [type_str]:
                param.pop("types")


def dump_schemas(output_dir: pathlib.Path, *, verbose: bool = True) -> None:
    """Dump all ONNX op schemas to YAML files under ``output_dir``.

    Files are written to ``<output_dir>/<domain>/{latest,old}/<Op>-<version>.yaml``.
    """
    schemas = sorted(
        onnx.defs.get_all_schemas_with_history(),
        key=lambda s: (s.domain, s.name, s.since_version),
    )
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    latest_versions: dict[tuple[str, str], int] = {}
    for schema in schemas:
        key = (schema.domain, schema.name)
        latest_versions[key] = max(
            latest_versions.get(key, schema.since_version), schema.since_version
        )

    for schema in schemas:
        schema_dict = dump_value(schema)
        _strip_derivable_param_types(schema_dict)
        domain = schema.domain or "ai.onnx"
        key = (schema.domain, schema.name)
        bucket = "latest" if latest_versions[key] == schema.since_version else "old"
        outdir = output_dir / domain / bucket
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{schema.name}-{schema.since_version}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            if verbose:
                print(f"Writing {path}")
            yaml.dump(schema_dict, f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Output ONNX spec in YAML format.")
    parser.add_argument(
        "--output", help="Output directory", required=True, type=pathlib.Path
    )
    args = parser.parse_args(argv)
    dump_schemas(args.output)


if __name__ == "__main__":
    main()
