# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined,arg-type"
"""Export the operator schemas of the installed `onnx` to spec files.

One-off migration tool. Writes `operator-specifications/<domain>/v<N>/<op>.md`.
A section that is equal to the previous version of the same operator is
omitted from the file and thereby inherited. The node determinism, output
counts and unchecked attributes are read from the properties exposed
temporarily for the migration.

Usage:
    python tools/schema_migration/export_specs.py [--domain DOMAIN ...]
"""

from __future__ import annotations

import argparse
import re
import textwrap
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w

import onnx
import onnx.defs
import onnx.helper
import onnx.printer

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS_DIR = REPO_ROOT / "operator-specifications"
SCHEMA_CC = REPO_ROOT / "onnx" / "defs" / "schema.cc"

_OPTION = {
    onnx.defs.OpSchema.FormalParameterOption.Single: None,
    onnx.defs.OpSchema.FormalParameterOption.Optional: "optional",
    onnx.defs.OpSchema.FormalParameterOption.Variadic: "variadic",
}
_DETERMINISM = {
    onnx.defs.OpSchema.NodeDeterminism.Unknown: "unknown",
    onnx.defs.OpSchema.NodeDeterminism.Deterministic: "deterministic",
    onnx.defs.OpSchema.NodeDeterminism.NonDeterministic: "non_deterministic",
}
# TOML comments of keys, by `(domain, name, since_version, key)`.
_COMMENTS = {
    ("ai.onnx", "SequenceMap", 17, "node_determinism"): (
        "Setting `NonDeterministic` since it needs to be evaluated after function inlining"
    ),
}
_DIFFERENTIATION = {
    onnx.defs.OpSchema.DifferentiationCategory.Unknown: None,
    onnx.defs.OpSchema.DifferentiationCategory.Differentiable: "differentiable",
    onnx.defs.OpSchema.DifferentiationCategory.NonDifferentiable: "non_differentiable",
}


def type_aliases() -> dict[frozenset[str], str]:
    """Return the `OpSchema` type helpers that consist of plain tensor types.

    Parsed from `schema.cc`; helpers composed in other ways are not considered.
    """
    pattern = re.compile(
        r"const std::vector<std::string>& OpSchema::(\w+)\(\) \{\s*"
        r"static const auto v =\s*types::Tensors\(\s*\{([^}]*)\}\);"
    )
    aliases: dict[frozenset[str], str] = {}
    for name, body in pattern.findall(SCHEMA_CC.read_text()):
        types = frozenset(
            f"tensor({t.lower()})" for t in re.findall(r"TensorProto::(\w+)", body)
        )
        aliases.setdefault(types, name)
    return aliases


def _param(p: onnx.defs.OpSchema.FormalParameter) -> dict[str, Any]:
    out: dict[str, Any] = {"name": p.name, "type": p.type_str, "desc": p.description}
    if (option := _OPTION[p.option]) is not None:
        out["option"] = option
    if not p.is_homogeneous:
        out["homogeneous"] = False
    if p.min_arity != 1:
        out["min_arity"] = p.min_arity
    if (diff := _DIFFERENTIATION[p.differentiation_category]) is not None:
        out["differentiation"] = diff
    return out


def _default(attr: onnx.defs.OpSchema.Attribute) -> Any:
    value = onnx.helper.get_attribute_value(attr.default_value)
    match attr.type:
        case onnx.defs.OpSchema.AttrType.FLOAT:
            return float(str(np.float32(value)))
        case onnx.defs.OpSchema.AttrType.FLOATS:
            return [float(str(np.float32(v))) for v in value]
        case onnx.defs.OpSchema.AttrType.STRING:
            return value.decode()
        case onnx.defs.OpSchema.AttrType.STRINGS:
            return [v.decode() for v in value]
        case onnx.defs.OpSchema.AttrType.INT | onnx.defs.OpSchema.AttrType.INTS:
            return value
    raise ValueError(f"unsupported default for attribute `{attr.name}`")


def _attribute(attr: onnx.defs.OpSchema.Attribute) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": str(attr.type).rsplit(".", 1)[-1],
        "desc": attr.description,
    }
    if attr.required:
        out["required"] = True
    elif attr.default_value.name:
        out["default"] = _default(attr)
    return out


def _types(allowed: list[str], aliases: dict[frozenset[str], str]) -> str | list[str]:
    return aliases.get(frozenset(allowed), list(allowed))


def _function_body(schema: onnx.defs.OpSchema, version: int) -> str:
    proto = onnx.FunctionProto()
    proto.ParseFromString(schema.get_function_with_opset_version(version))
    imports = [(o.domain, o.version) for o in proto.opset_import]
    if imports != [(schema.domain, version)]:
        raise ValueError(
            f"unsupported opset imports `{imports}` in function body of "
            f"`{schema.name}` version `{version}`"
        )
    text = onnx.printer.to_text(proto)
    # Keep only the node list, i.e., drop the function header.
    return text[text.index("\n{\n") + 1 :] + "\n"


def to_spec(
    schema: onnx.defs.OpSchema, aliases: dict[frozenset[str], str]
) -> dict[str, Any]:
    """Return the fully resolved spec of `schema`, including the doc string."""
    spec: dict[str, Any] = {"name": schema.name, "domain": schema.domain or "ai.onnx"}
    spec["support_level"] = (
        "experimental"
        if schema.support_level == onnx.defs.OpSchema.SupportType.EXPERIMENTAL
        else "common"
    )
    spec["deprecated"] = schema.deprecated
    spec["node_determinism"] = _DETERMINISM[schema._declared_node_determinism]
    if schema._allowed_output_nums is not None:
        spec["num_outputs"] = schema._allowed_output_nums
    spec["allow_unchecked_attributes"] = schema._allows_unchecked_attributes
    spec["attributes"] = {
        name: _attribute(a) for name, a in sorted(schema.attributes.items())
    }
    spec["inputs"] = [_param(p) for p in schema.inputs]
    spec["outputs"] = [_param(p) for p in schema.outputs]
    spec["type_constraints"] = {
        tc.type_param_str: {
            "types": _types(tc.allowed_type_strs, aliases),
            "desc": tc.description,
        }
        for tc in schema.type_constraints
    }
    spec["function_bodies"] = {
        str(v): _function_body(schema, v) for v in schema.function_opset_versions
    }
    spec["doc"] = _dedent(schema.doc or "")
    return spec


def _dedent(doc: str) -> str:
    """Remove the common indentation of all lines but the first and trailing whitespace."""
    first, _, rest = doc.strip().partition("\n")
    lines = f"{first}\n{textwrap.dedent(rest)}".strip().split("\n")
    return "\n".join(line.rstrip() for line in lines)


# Maximum length of a line with an array before it is split into one line per item.
_MAX_LINE = 100


def _key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else _scalar(key)


def _scalar(value: Any) -> str:
    return tomli_w.dumps({"v": value}, multiline_strings=True).removeprefix("v = ")[:-1]


def _key_value(key: str, value: Any) -> str:
    prefix = f"{_key(key)} = "
    if isinstance(value, dict):
        if value:
            raise ValueError(f"unexpected non-empty table `{key}`")
        return prefix + "{}"
    if not isinstance(value, list):
        return prefix + _scalar(value)
    items = [_scalar(v) for v in value]
    line = prefix + "[" + ", ".join(items) + "]"
    if len(line) <= _MAX_LINE:
        return line
    return prefix + "[\n" + "".join(f"    {i},\n" for i in items) + "]"


def to_toml(data: dict[str, Any], comments: dict[str, str]) -> str:
    """Return `data` as TOML 1.0, with `comments` preceding their keys.

    Inputs, outputs, attributes, type constraints and function bodies are
    always written as (arrays of) tables; all other keys precede them.
    """
    top: list[str] = []
    tables: list[list[str]] = []
    for key, value in data.items():
        if key in ("inputs", "outputs") and value:
            tables += [
                [f"[[{key}]]"] + [_key_value(k, v) for k, v in p.items()] for p in value
            ]
        elif key in ("attributes", "type_constraints") and value:
            tables += [
                [f"[{key}.{_key(name)}]"] + [_key_value(k, v) for k, v in t.items()]
                for name, t in value.items()
            ]
        elif key == "function_bodies" and value:
            tables.append([f"[{key}]"] + [_key_value(k, v) for k, v in value.items()])
        else:
            if key in comments:
                top.append(f"# {comments[key]}")
            top.append(_key_value(key, value))
    toml = "\n\n".join("\n".join(lines) for lines in [top, *tables]) + "\n"
    if tomllib.loads(toml) != data:
        raise ValueError(f"TOML does not round-trip:\n{toml}")
    return toml


def render(
    spec: dict[str, Any], previous: dict[str, Any] | None, comments: dict[str, str]
) -> str:
    """Render `spec` as a spec file, omitting sections equal to `previous`."""
    previous = previous or {}
    if unset := set(previous) - set(spec):
        raise ValueError(f"`{spec['name']}` unsets the inherited keys {unset}")
    data = {
        key: value
        for key, value in spec.items()
        if key in _NOT_INHERITED
        or (key != "doc" and value != previous.get(key, _EMPTY.get(key)))
    }
    doc = spec["doc"] if spec["doc"] != previous.get("doc", "") else ""
    toml = to_toml(data, comments)
    # The fence must be longer than any backtick fence within the TOML strings.
    runs = re.findall(r"^`{3,}", toml, re.MULTILINE)
    fence = "`" * max([3] + [len(r) + 1 for r in runs])
    out = f"{fence}toml\n{toml}{fence}\n"
    if doc:
        out += f"\n{doc}\n"
    return out


# Keys that are stated in every file.
_NOT_INHERITED = ("name", "domain")

# Values that are equivalent to an absent key in the first version.
_EMPTY: dict[str, Any] = {
    "support_level": "common",
    "deprecated": False,
    "node_determinism": "unknown",
    "allow_unchecked_attributes": False,
    "attributes": {},
    "inputs": [],
    "outputs": [],
    "type_constraints": {},
    "function_bodies": {},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        action="append",
        help="only export operators of this domain (may be repeated)",
    )
    args = parser.parse_args()
    aliases = type_aliases()
    by_op: dict[tuple[str, str], list[onnx.defs.OpSchema]] = defaultdict(list)
    for schema in onnx.defs.get_all_schemas_with_history():
        domain = schema.domain or "ai.onnx"
        if args.domain is None or domain in args.domain:
            by_op[domain, schema.name].append(schema)

    names = defaultdict(set)
    count = 0
    for (domain, name), schemas in sorted(by_op.items()):
        names[domain, name.lower()].add(name)
        if len(names[domain, name.lower()]) > 1:
            raise ValueError(f"operator names differ only in case: `{name}`")
        previous = None
        for schema in sorted(schemas, key=lambda s: s.since_version):
            spec = to_spec(schema, aliases)
            comments = {
                key: comment
                for (d, n, v, key), comment in _COMMENTS.items()
                if (d, n, v) == (domain, name, schema.since_version)
            }
            path = (
                SPECS_DIR / domain / f"v{schema.since_version}" / f"{name.lower()}.md"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(spec, previous, comments))
            previous = spec
            count += 1
    print(f"wrote {count} spec files to {SPECS_DIR}")


if __name__ == "__main__":
    main()
