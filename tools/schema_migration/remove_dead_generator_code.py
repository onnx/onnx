# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr,arg-type,return-value"
"""Remove the code of schema generators that no longer affects the schema.

One-off migration tool, to be run after 'remove_empty_generators.py'. The
statements 'POPULATE_OP_DOC_STR(...);', which only build doc strings, are
removed. Within
the body of a generator and of the lambda it returns, a statement is kept if it
uses the 'OpSchema&' or an identifier used by another kept statement; all other
statements, such as those building doc strings, are removed, as are the
captures of the lambda it no longer uses. Parameters of a generator that are no
longer used are removed together with the corresponding arguments of all calls.

Requires 'tree-sitter' and 'tree-sitter-cpp'. Run 'pixi run lint' afterwards.

Usage:
    python tools/schema_migration/remove_dead_generator_code.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from registrations import DEFS_DIR, PARSER, Generator, generators, walk

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node


def files() -> list[Path]:
    return sorted(DEFS_DIR.glob("*/*.cc")) + sorted(DEFS_DIR.glob("*/*.h"))


def _skip_literal(src: bytes, i: int) -> int:
    """Return the index after the string or character literal starting at `i`."""
    if src.startswith(b'R"', i):
        open_paren = src.index(b"(", i)
        delimiter = b")" + src[i + 2 : open_paren] + b'"'
        return src.index(delimiter, open_paren) + len(delimiter)
    quote = src[i : i + 1]
    i += 1
    while src[i : i + 1] != quote:
        i += 2 if src[i : i + 1] == b"\\" else 1
    return i + 1


def remove_doc_populators(src: bytes) -> bytes:
    """Remove all statements `POPULATE_OP_DOC_STR(...);`.

    The macro only populates doc strings, which the spec files contain.
    tree-sitter does not parse its argument, a sequence of statements.
    """
    macro = b"POPULATE_OP_DOC_STR("
    while (start := src.find(macro)) != -1:
        i, depth = start + len(macro), 1
        while depth:
            if src[i : i + 2] == b'R"' or src[i : i + 1] in (b'"', b"'"):
                i = _skip_literal(src, i)
                continue
            depth += {b"(": 1, b")": -1}.get(src[i : i + 1], 0)
            i += 1
        if src[i : i + 1] == b";":
            i += 1
        line_start = src.rfind(b"\n", 0, start) + 1
        if not src[line_start:start].strip() and src[i : i + 1] == b"\n":
            start, i = line_start, i + 1
        src = src[:start] + src[i:]
    return src


def identifiers(node: Node) -> set[bytes]:
    return {n.text for n in walk(node) if n.type == "identifier"}


def dead_statements(body: Node, live: set[bytes]) -> list[Node]:
    """Return the statements of `body` that use no identifier in `live`.

    `live` is extended by the identifiers of the kept statements.
    """
    statements = [c for c in body.named_children if c.type != "comment"]
    kept: list[Node] = []
    changed = True
    while changed:
        changed = False
        for statement in statements:
            if statement not in kept and identifiers(statement) & live:
                kept.append(statement)
                live |= identifiers(statement)
                changed = True
    return [s for s in statements if s not in kept]


def returned_lambda(body: Node) -> Node | None:
    for statement in body.named_children:
        if statement.type == "return_statement":
            for node in statement.named_children:
                if node.type == "lambda_expression":
                    return node
    return None


def dead_spans(gen: Generator) -> list[tuple[int, int]]:
    body = gen.node.child_by_field_name("body")
    lam = returned_lambda(body)
    if lam is None:
        return []
    live = {b"schema"}
    dead = dead_statements(lam.child_by_field_name("body"), live)
    # The returned lambda uses `live`; the statements before it compute them.
    dead += dead_statements(body, live | identifiers(lam))
    return [(s.start_byte, s.end_byte) for s in dead]


def unused_capture_spans(gen: Generator) -> list[tuple[int, int]]:
    """Return the spans of the captures that the returned lambda does not use."""
    lam = returned_lambda(gen.node.child_by_field_name("body"))
    if lam is None:
        return []
    used = identifiers(lam.child_by_field_name("body"))
    captures = lam.child_by_field_name("captures")
    unused = set()
    for i, capture in enumerate(list_items(captures)):
        if capture.type == "identifier":
            name = capture.text
        elif capture.type == "lambda_capture_initializer":
            name = capture.named_children[0].text
        else:
            continue
        if name not in used:
            unused.add(i)
    return list_item_spans(captures, unused)


def parameter_list(gen: Generator) -> Node:
    declarator = gen.node.child_by_field_name("declarator")
    while declarator.type != "function_declarator":
        declarator = declarator.child_by_field_name("declarator")
    return declarator.child_by_field_name("parameters")


def list_items(node: Node) -> list[Node]:
    """Return the items of a parameter, argument or capture list."""
    return [c for c in node.named_children if c.type != "comment"]


def parameter_name(param: Node) -> bytes | None:
    declarator = param.child_by_field_name("declarator")
    if declarator is None:
        return None
    return next((n.text for n in walk(declarator) if n.type == "identifier"), None)


def remove_spans(src: bytes, spans: list[tuple[int, int]]) -> bytes:
    for start, end in sorted(spans, reverse=True):
        src = src[:start] + src[end:]
    return src


def list_item_spans(node: Node, indices: set[int]) -> list[tuple[int, int]]:
    """Return the spans of the items `indices` of a list, including separating commas.

    Comments following a removed item are removed with it.
    """
    items = list_items(node)
    closing = node.children[-1].start_byte
    spans = []
    removed = sorted(i for i in indices if i < len(items))
    # Maximal runs `first..last` of consecutive removed items.
    runs: list[list[int]] = []
    for i in removed:
        if runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    for first, last in runs:
        if last + 1 < len(items):
            spans.append((_start(items[first]), _start(items[last + 1])))
        elif first > 0:
            comma = next(
                c
                for c in node.children
                if c.type == "," and c.start_byte >= items[first - 1].end_byte
            )
            spans.append((comma.start_byte, closing))
        else:
            spans.append((_start(items[first]), closing))
    return spans


def _start(item: Node) -> int:
    """Return the start of a list item, including a capture by reference `&`."""
    prev = item.prev_sibling
    return prev.start_byte if prev is not None and prev.type == "&" else item.start_byte


def unused_parameters() -> dict[str, set[int]]:
    """Return the indices of the unused parameters of every generator."""
    out = {}
    for name, gen in generators().items():
        body = gen.node.child_by_field_name("body")
        used = identifiers(body)
        # Unnamed parameters are unused as well.
        unused = {
            i
            for i, p in enumerate(list_items(parameter_list(gen)))
            if parameter_name(p) not in used
        }
        if unused:
            out[name] = unused
    return out


def remove_parameters(unused: dict[str, set[int]]) -> None:
    """Remove the parameters from all declarations, definitions and calls."""
    for path in files():
        src = path.read_bytes()
        spans = []
        for node in walk(PARSER.parse(src).root_node):
            if node.type == "function_declarator":
                name = node.child_by_field_name("declarator").text.decode()
                if name in unused:
                    parameters = node.child_by_field_name("parameters")
                    spans += list_item_spans(parameters, unused[name])
            elif node.type == "call_expression":
                callee = node.child_by_field_name("function").text.decode()
                if (name := callee.rsplit("::", 1)[-1]) in unused:
                    arguments = node.child_by_field_name("arguments")
                    spans += list_item_spans(arguments, unused[name])
        if spans:
            path.write_bytes(remove_spans(src, spans))


def remove_dead_statements() -> int:
    by_path: dict[Path, list[tuple[int, int]]] = {}
    for gen in generators().values():
        by_path.setdefault(gen.path, []).extend(dead_spans(gen))
    for path, spans in by_path.items():
        if spans:
            path.write_bytes(remove_spans(path.read_bytes(), spans))
    count = sum(len(spans) for spans in by_path.values())
    by_path = {}
    for gen in generators().values():
        by_path.setdefault(gen.path, []).extend(unused_capture_spans(gen))
    for path, spans in by_path.items():
        if spans:
            path.write_bytes(remove_spans(path.read_bytes(), spans))
    return count + sum(len(spans) for spans in by_path.values())


def main() -> None:
    for path in files():
        src = path.read_bytes()
        if (new := remove_doc_populators(src)) != src:
            path.write_bytes(new)
    statements = parameters_removed = 0
    while True:
        statements += remove_dead_statements()
        if not (unused := unused_parameters()):
            break
        remove_parameters(unused)
        parameters_removed += sum(len(i) for i in unused.values())
    print(f"removed {statements} statements and {parameters_removed} parameters")


if __name__ == "__main__":
    main()
