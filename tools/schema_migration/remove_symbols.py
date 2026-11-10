# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr"
"""Remove top-level declarations, aliases and function definitions by name.

Shared by the migration scripts. A comment block directly preceding a removed
symbol and a comment following it on the same line are removed as well.
"""

from __future__ import annotations

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

PARSER = Parser(Language(tree_sitter_cpp.language()))
_NESTING = (
    "namespace_definition",
    "declaration_list",
    "preproc_ifdef",
    "preproc_if",
    "preproc_else",
)


def top_level(node: Node):
    for child in node.children:
        if child.type in _NESTING:
            yield from top_level(child)
        else:
            yield child


def declared_name(node: Node) -> str | None:
    if node.type == "alias_declaration":
        return node.child_by_field_name("name").text.decode()
    declarator = node.child_by_field_name("declarator")
    while declarator is not None and declarator.type not in (
        "identifier",
        "qualified_identifier",
    ):
        declarator = declarator.child_by_field_name("declarator")
    return declarator.text.decode() if declarator is not None else None


def remove(src: bytes, names: set[str]) -> tuple[bytes, set[str]]:
    """Remove the top-level symbols `names` from `src`; return it and the names found."""
    spans = []
    found = set()
    for node in top_level(PARSER.parse(src).root_node):
        name = (
            declared_name(node)
            if node.type in ("declaration", "function_definition", "alias_declaration")
            else None
        )
        if name not in names:
            continue
        start, line, prev = node.start_byte, node.start_point[0], node.prev_sibling
        while (
            prev is not None
            and prev.type == "comment"
            and prev.end_point[0] == line - 1
        ):
            start, line, prev = prev.start_byte, prev.start_point[0], prev.prev_sibling
        end = node.end_byte
        following = node.next_sibling
        if (
            following is not None
            and following.type == "comment"
            and following.start_point[0] == node.end_point[0]
        ):
            end = following.end_byte
        while src[end : end + 1] == b"\n":
            end += 1
        spans.append((start, end))
        found.add(name)
    for start, end in sorted(spans, reverse=True):
        src = src[:start] + src[end:]
    return src, found
