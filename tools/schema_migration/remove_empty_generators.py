# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr,arg-type,assignment"
"""Remove schema generators that no longer modify the schema.

One-off migration tool, to be run after 'rewrite_registrations.py'. A
generator is empty if its body no longer uses the 'OpSchema&' it fills or if it
only forwards to an empty generator.
Empty generators are removed together with their declarations and all
'.FillUsing(Generator(...))' calls. Remaining generators are reported.

Requires 'tree-sitter' and 'tree-sitter-cpp'. Run 'pixi run lint' afterwards.

Usage:
    python tools/schema_migration/remove_empty_generators.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from registrations import (
    DEFS_DIR,
    PARSER,
    Generator,
    arguments,
    generators,
    location,
    method_name,
    walk,
)
from remove_symbols import remove

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node


def files() -> list[Path]:
    return sorted(DEFS_DIR.glob("*/*.cc")) + sorted(DEFS_DIR.glob("*/*.h"))


def in_parameter_list(node: Node) -> bool:
    while node is not None:
        if node.type == "parameter_declaration":
            return True
        node = node.parent
    return False


def uses_schema(gen: Generator) -> bool:
    body = gen.node.child_by_field_name("body")
    return any(
        n.type == "identifier" and n.text == b"schema" and not in_parameter_list(n)
        for n in walk(body)
    )


def empty_generators(gens: dict[str, Generator]) -> set[str]:
    empty = {
        name
        for name, gen in gens.items()
        if gen.forward is None and not uses_schema(gen)
    }
    forwards = {name: gen for name, gen in gens.items() if gen.forward is not None}
    while True:
        new = {
            name
            for name, gen in forwards.items()
            if name not in empty
            and gen.forward.child_by_field_name("function")
            .text.decode()
            .rsplit("::", 1)[-1]
            in empty
        }
        if not new:
            return empty
        empty |= new


def remove_fill_using(src: bytes, empty: set[str]) -> bytes:
    """Remove `.FillUsing(Generator(...))` of the empty generators from chains."""
    spans = []
    for node in walk(PARSER.parse(src).root_node):
        if node.type != "call_expression" or method_name(node) != b"FillUsing":
            continue
        (arg,) = arguments(node)
        callee = arg.child_by_field_name("function").text.decode()
        if callee.rsplit("::", 1)[-1] in empty:
            receiver = node.child_by_field_name("function").child_by_field_name(
                "argument"
            )
            spans.append((receiver.end_byte, node.end_byte))
    for start, end in sorted(spans, reverse=True):
        src = src[:start] + src[end:]
    return src


def main() -> None:
    gens = generators()
    empty = empty_generators(gens)
    for path in files():
        src = remove_fill_using(path.read_bytes(), empty)
        src, _ = remove(src, empty)
        path.write_bytes(src)
    print(f"removed {len(empty)} generators")
    for name in sorted(set(gens) - empty):
        print(f"{location(gens[name].path, gens[name].node)}: kept `{name}`")


if __name__ == "__main__":
    main()
