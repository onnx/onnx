# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr,assignment"
"""Rewrite operator registrations to start from the generated schema functions.

One-off migration tool, to be run after 'export_specs.py'. In every
registration in 'onnx/defs/*/*.cc', 'OpSchema()' is replaced by
'generated::<domain>::<Name>_ver<N>()' and the builder calls whose
information is contained in the spec files are removed. In schema generators,
i.e., in statements 'schema.X(...);', these calls are removed as well. Any
other call than those in 'KEPT' is reported as an error.

Requires 'tree-sitter' and 'tree-sitter-cpp'. Run 'pixi run lint' afterwards.

Usage:
    python tools/schema_migration/rewrite_registrations.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from registrations import (
    PARSER,
    REPO_ROOT,
    Registration,
    location,
    method_name,
    registrations,
    schema_statements,
    source_files,
    unroll_chain,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node

# Builder calls whose information is contained in the spec files.
DECLARATIVE = {
    b"SetDoc",
    b"Attr",
    b"Input",
    b"Output",
    b"TypeConstraint",
    b"SetSupportLevel",
    b"Deprecate",
    b"FunctionBody",
    b"SetNodeDeterminism",
    b"NumOutputs",
    b"AllowUncheckedAttributes",
}

# Calls that remain in C++.
KEPT = {
    b"TypeAndShapeInferenceFunction",
    b"PartialDataPropagationFunction",
    b"SetContextDependentFunctionBodyBuilder",
    b"FillUsing",
    # Called on the schema passed to context-dependent function body builders.
    b"BuildFunction",
}

INCLUDE = b'#include "onnx/defs/generated/schemas.h"\n'


@dataclass
class Edit:
    start: int
    end: int
    text: bytes


def method_text(src: bytes, call: Node) -> bytes:
    """Return `.Name(args)` of a method call in a chain, with preceding comments."""
    receiver = call.child_by_field_name("function").child_by_field_name("argument")
    return src[receiver.end_byte : call.end_byte]


def unknown_calls(path: Path, calls: list[Node]) -> list[str]:
    return [
        f"{location(path, c)}: unknown builder call `{method_name(c).decode()}`"
        for c in calls
        if method_name(c) not in DECLARATIVE | KEPT
    ]


def kept_calls(src: bytes, calls: list[Node]) -> bytes:
    return b"".join(
        method_text(src, c) for c in calls if method_name(c) not in DECLARATIVE
    )


def rewrite_registration(src: bytes, reg: Registration) -> list[Edit]:
    namespace = reg.domain.replace(".", "_")
    base = f"generated::{namespace}::{reg.name}_ver{reg.version}()".encode()
    expr = reg.calls[-1] if reg.calls else reg.base
    edits = [
        Edit(reg.base.start_byte, expr.end_byte, base + kept_calls(src, reg.calls))
    ]
    if reg.calls and method_name(reg.calls[-1]) in DECLARATIVE:
        # A comment after the registration describes its removed last call.
        statement = reg.node
        while statement.parent.type not in ("translation_unit", "declaration_list"):
            statement = statement.parent
        comment = statement.next_sibling
        if (
            comment is not None
            and comment.type == "comment"
            and comment.start_point[0] == statement.end_point[0]
        ):
            edits.append(Edit(statement.end_byte, comment.end_byte, b""))
    return edits


def rewrite_statement(src: bytes, statement: Node) -> Edit | None:
    """Remove the declarative calls from a statement `schema.X(...)...;`."""
    _, calls = unroll_chain(statement.named_children[0])
    if all(method_name(c) not in DECLARATIVE for c in calls):
        return None
    start, end = statement.start_byte, statement.end_byte
    if kept := kept_calls(src, calls):
        return Edit(start, end, b"schema" + kept + b";")
    if statement.parent.type != "compound_statement":
        # The body of a control statement without braces.
        return Edit(start, end, b"{}")
    # Remove the whole line if the statement is the only thing on it.
    line_start = src.rfind(b"\n", 0, start) + 1
    if not src[line_start:start].strip() and src[end : end + 1] == b"\n":
        start, end = line_start, end + 1
    return Edit(start, end, b"")


def check_file(path: Path) -> list[str]:
    root = PARSER.parse(path.read_bytes()).root_node
    errors = [
        e for reg in registrations(path, root) for e in unknown_calls(path, reg.calls)
    ]
    for statement in schema_statements(root):
        _, calls = unroll_chain(statement.named_children[0])
        errors += unknown_calls(path, calls)
    return errors


def rewrite_file(path: Path) -> int:
    src = path.read_bytes()
    root = PARSER.parse(src).root_node
    regs = registrations(path, root)
    statements = schema_statements(root)
    edits = [e for reg in regs for e in rewrite_registration(src, reg)]
    edits += [e for s in statements if (e := rewrite_statement(src, s))]
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        src = src[: edit.start] + edit.text + src[edit.end :]
    if regs and INCLUDE not in src:
        first = src.index(b'#include "onnx/')
        src = src[:first] + INCLUDE + src[first:]
    path.write_bytes(src)
    return len(regs)


def main() -> None:
    if errors := [e for path in source_files() for e in check_file(path)]:
        sys.exit("\n".join(errors))
    total = 0
    for path in source_files():
        if n := rewrite_file(path):
            print(f"{path.relative_to(REPO_ROOT)}: {n} registrations")
            total += n
    print(f"rewrote {total} registrations")


if __name__ == "__main__":
    main()
