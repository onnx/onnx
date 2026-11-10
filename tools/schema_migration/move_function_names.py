# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr,arg-type,assignment"
"""Move the functions set by the registrations to the spec files.

One-off migration tool, to be run after 'move_functions_to_utils.py'. The
functions passed to 'TypeAndShapeInferenceFunction',
'PartialDataPropagationFunction' and 'SetContextDependentFunctionBodyBuilder'
by a registration, directly or via '.FillUsing(Generator(...))', are written
to the keys 'shape_inference', 'data_propagation' and 'function_body_builders'
of its spec file. The calls and the '.FillUsing(...)' calls are removed from
the registrations. Comments preceding a removed call become TOML comments.
Schema generators that set other parts of the schema or a context-dependent
function body builder are reported as errors.

Requires 'tree-sitter' and 'tree-sitter-cpp'. Run 'pixi run lint' afterwards.

Usage:
    python tools/schema_migration/move_function_names.py
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

from registrations import (
    DATA_PROPAGATION,
    PARSER,
    REPO_ROOT,
    SHAPE_INFERENCE,
    Generator,
    RegistrationError,
    arguments,
    called_generator,
    chain_functions,
    function_name,
    generators,
    location,
    method_name,
    registrations,
    source_files,
    unroll_chain,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node

SPECS_DIR = REPO_ROOT / "operator-specifications"
BUILDER = b"SetContextDependentFunctionBodyBuilder"
FILL_USING = b"FillUsing"
# Keys whose functions are set by a method.
KEYS = {
    SHAPE_INFERENCE: "shape_inference",
    DATA_PROPAGATION: "data_propagation",
    BUILDER: "function_body_builders",
    FILL_USING: "shape_inference",
}


def check_generator(gen: Generator, gens: dict[str, Generator]) -> None:
    """Raise unless `gen` only sets inference and data propagation functions."""
    if gen.forward is not None:
        check_generator(called_generator(gen.forward, gen.path, gens), gens)
        return
    for statement in gen.statements:
        _, calls = unroll_chain(statement.named_children[0])
        for call in calls:
            if method_name(call) not in (SHAPE_INFERENCE, DATA_PROPAGATION):
                raise RegistrationError(
                    f"{location(gen.path, call)}: unsupported call in `{gen.name}`"
                )


def preceding_comments(src: bytes, start: int, call: Node) -> list[str]:
    """Return the lines of the C++ comments between `start` and the method call."""
    end = call.child_by_field_name("function").child_by_field_name("field").start_byte
    return [m.strip() for m in re.findall(r"//(.*)", src[start:end].decode())]


def toml_value(value: Any) -> str:
    if value is False:
        return "false"
    if isinstance(value, dict):
        items = ", ".join(f'{k} = "{v}"' for k, v in sorted(value.items()))
        return f"{{ {items} }}"
    return f'"{value}"'


def write_keys(
    path: Path, keys: dict[str, Any], comments: dict[str, list[str]]
) -> None:
    """Append `keys` with their comments to the top-level keys of a spec file."""
    text = path.read_text()
    fence = re.match(r"`{3,}", text).group()
    end = text.index(f"\n{fence}\n", len(fence)) + 1
    toml = text[:end]
    table = re.search(r"^\[", toml, re.MULTILINE)
    position = table.start() - 1 if table else end
    lines = "".join(
        "".join(f"# {c}\n" for c in comments.get(k, [])) + f"{k} = {toml_value(v)}\n"
        for k, v in keys.items()
    )
    path.write_text(text[:position] + lines + text[position:])


def main() -> None:
    gens = generators()
    count = 0
    try:
        for path in source_files():
            src = path.read_bytes()
            spans = []
            for reg in registrations(path, PARSER.parse(src).root_node):
                functions = chain_functions(reg.calls, path, gens)
                keys: dict[str, Any] = {
                    "shape_inference": functions.shape_inference or False
                }
                if functions.data_propagation:
                    keys["data_propagation"] = functions.data_propagation
                builders: dict[int, str] = {}
                comments: dict[str, list[str]] = {}
                for call in reg.calls:
                    name = method_name(call)
                    if name == FILL_USING:
                        check_generator(
                            called_generator(arguments(call)[0], path, gens), gens
                        )
                    elif name == BUILDER:
                        args = arguments(call)
                        opset = int(args[1].text) if len(args) > 1 else reg.version
                        builders[opset] = function_name(args[0], path)
                    elif name not in (SHAPE_INFERENCE, DATA_PROPAGATION):
                        continue
                    receiver = call.child_by_field_name("function").child_by_field_name(
                        "argument"
                    )
                    comments.setdefault(KEYS[name], []).extend(
                        preceding_comments(src, receiver.end_byte, call)
                    )
                    spans.append((receiver.end_byte, call.end_byte))
                if builders:
                    keys["function_body_builders"] = builders
                spec = (
                    SPECS_DIR
                    / reg.domain
                    / f"v{reg.version}"
                    / f"{reg.name.lower()}.md"
                )
                write_keys(spec, keys, comments)
                count += 1
            for start, end in sorted(spans, reverse=True):
                src = src[:start] + src[end:]
            path.write_bytes(src)
    except RegistrationError as e:
        sys.exit(str(e))
    print(f"updated {count} spec files")


if __name__ == "__main__":
    main()
