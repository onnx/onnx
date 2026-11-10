# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined,arg-type"
"""Dump all operator schemas of the installed `onnx` to JSON.

Used to verify that the migration to spec files leaves the registry unchanged.
Only the order of inputs and outputs is considered significant; attributes and
type constraints are sorted.

Usage:
    python tools/schema_migration/dump_registry.py OUTPUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx
import onnx.defs
import onnx.printer


def _param(p: onnx.defs.OpSchema.FormalParameter) -> dict[str, Any]:
    return {
        "name": p.name,
        "type_str": p.type_str,
        "description": p.description,
        "option": str(p.option),
        "is_homogeneous": p.is_homogeneous,
        "min_arity": p.min_arity,
        "differentiation_category": str(p.differentiation_category),
    }


def _function(serialized: bytes) -> str:
    proto = onnx.FunctionProto()
    proto.ParseFromString(serialized)
    return onnx.printer.to_text(proto)


def dump_schema(s: onnx.defs.OpSchema) -> dict[str, Any]:
    return {
        "name": s.name,
        "domain": s.domain,
        "since_version": s.since_version,
        "doc": s.doc,
        "support_level": str(s.support_level),
        "deprecated": s.deprecated,
        "node_determinism": str(s.node_determinism),
        "min_input": s.min_input,
        "max_input": s.max_input,
        "min_output": s.min_output,
        "max_output": s.max_output,
        "inputs": [_param(p) for p in s.inputs],
        "outputs": [_param(p) for p in s.outputs],
        "attributes": {
            name: {
                "type": str(a.type),
                "description": a.description,
                "required": a.required,
                "default_value": str(a.default_value),
            }
            for name, a in sorted(s.attributes.items())
        },
        "type_constraints": {
            tc.type_param_str: {
                "description": tc.description,
                "allowed_type_strs": sorted(tc.allowed_type_strs),
            }
            for tc in sorted(s.type_constraints, key=lambda tc: tc.type_param_str)
        },
        "functions": {
            str(v): _function(s.get_function_with_opset_version(v))
            for v in s.function_opset_versions
        },
        "context_dependent_function_opset_versions": list(
            s.context_dependent_function_opset_versions
        ),
        "has_type_and_shape_inference_function": s.has_type_and_shape_inference_function,
        "has_data_propagation_function": s.has_data_propagation_function,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    schemas = sorted(
        onnx.defs.get_all_schemas_with_history(),
        key=lambda s: (s.domain, s.name, s.since_version),
    )
    dump = {
        f"{s.domain or 'ai.onnx'}/{s.name}/{s.since_version}": dump_schema(s)
        for s in schemas
    }
    args.output.write_text(json.dumps(dump, indent=1, sort_keys=False) + "\n")
    print(f"wrote {len(dump)} schemas to {args.output}")


if __name__ == "__main__":
    main()
