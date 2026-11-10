# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr,arg-type,assignment"
"""Replace the lambdas passed to the function setters of schemas by named functions.

One-off migration tool. Every lambda passed to 'TypeAndShapeInferenceFunction',
'PartialDataPropagationFunction' or 'SetContextDependentFunctionBodyBuilder' in
a registration or schema generator in 'onnx/defs/*/*.cc' is replaced as follows:

- A lambda that only forwards its argument to a function is replaced by that
  function.
- Otherwise, it is replaced by a 'static' function with the lambda's body,
  defined before the registration or generator. In a registration of '<Op>'
  with version 'N', it is named '<Op>ShapeInference_opset<N>',
  '<Op>DataPropagation_opset<N>' or '<Op>FunctionBody_opset<M>', where 'M' is
  the opset version of the function body. In a generator '<Name>Generator<S>',
  where '<Name>' does not end with 'OpSchema', 'Doc' or 'Op', it is named
  '<Name>ShapeInference<S>' etc.

Lambdas with captures are reported as errors.

Requires 'tree-sitter' and 'tree-sitter-cpp'. Run 'pixi run lint' afterwards.

Usage:
    python tools/schema_migration/name_functions.py
"""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

from registrations import (
    PARSER,
    REPO_ROOT,
    arguments,
    generator_name,
    is_generator,
    location,
    method_name,
    registrations,
    source_files,
    unroll_chain,
    walk,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tree_sitter import Node

# Setters by name: the kind used in function names and the return type.
SETTERS = {
    b"TypeAndShapeInferenceFunction": ("ShapeInference", "void"),
    b"PartialDataPropagationFunction": ("DataPropagation", "void"),
    b"SetContextDependentFunctionBodyBuilder": ("FunctionBody", "bool"),
}


class NamingError(Exception):
    pass


def generator_stem(name: str) -> tuple[str, str]:
    """Split a generator name into the name of its functions and a suffix."""
    match = re.fullmatch(r"(\w+?)(?:OpSchema|Doc|Op)?Generator(\w*)", name)
    if match is None:
        raise NamingError(f"unexpected generator name `{name}`")
    return match.group(1), match.group(2)


def forwarded(lam: Node) -> str | None:
    """Return `f` if the body of `lam` is `{ f(<params>); }` or `{ return f(<params>); }`."""
    declarator = lam.child_by_field_name("declarator")
    params = declarator.child_by_field_name("parameters")
    names = [
        n.text.decode()
        for p in params.named_children
        for n in walk(p.child_by_field_name("declarator") or p)
        if n.type == "identifier"
    ]
    body = re.sub(r"\s+", " ", lam.child_by_field_name("body").text.decode())
    match = re.fullmatch(r"\{ (?:return )?([\w:]+)\(([^()]*)\); \}", body)
    if match and [a.strip() for a in match.group(2).split(",")] == names:
        return match.group(1)
    return None


def definition(name: str, return_type: str, lam: Node) -> str:
    declarator = lam.child_by_field_name("declarator")
    params = declarator.child_by_field_name("parameters").text.decode()
    body = lam.child_by_field_name("body").text.decode()
    return f"static {return_type} {name}{params} {body}\n\n"


def start_of_line(src: bytes, node: Node) -> int:
    """Return the start of the line of `node`, including directly preceding comments."""
    prev = node.prev_sibling
    while (
        prev is not None
        and prev.type == "comment"
        and prev.end_point[0] + 1 >= node.start_point[0]
    ):
        node, prev = prev, prev.prev_sibling
    return src.rfind(b"\n", 0, node.start_byte) + 1


def statement_of(node: Node) -> Node:
    while node.parent.type not in ("translation_unit", "declaration_list"):
        node = node.parent
    return node


def lambda_argument(call: Node, path: Path) -> Node | None:
    args = arguments(call)
    if not args or args[0].type != "lambda_expression":
        return None
    lam = args[0]
    if lam.child_by_field_name("captures").text != b"[]":
        raise NamingError(f"{location(path, lam)}: lambda with captures")
    return lam


def opset_of(call: Node, version: int, path: Path) -> int:
    args = arguments(call)
    if len(args) < 2:  # noqa: PLR2004
        return version
    if args[1].type != "number_literal":
        raise NamingError(f"{location(path, call)}: opset version is not a literal")
    return int(args[1].text)


def name_file(path: Path, taken: set[str]) -> int:
    """Name the lambdas in `path`; `taken` holds the names in use."""
    src = path.read_bytes()
    root = PARSER.parse(src).root_node
    # (lambda, name, position of the definition, return type)
    found: list[tuple[Node, str, int, str]] = []
    for reg in registrations(path, root):
        for call in reg.calls:
            if (setter := SETTERS.get(method_name(call))) is None:
                continue
            if (lam := lambda_argument(call, path)) is None:
                continue
            kind, return_type = setter
            version = reg.version
            if kind == "FunctionBody":
                version = opset_of(call, reg.version, path)
            name = f"{reg.name}{kind}_opset{version}"
            position = start_of_line(src, statement_of(reg.node))
            found.append((lam, name, position, return_type))
    for node in walk(root):
        if not is_generator(node):
            continue
        stem, suffix = generator_stem(generator_name(node))
        for call in walk(node.child_by_field_name("body")):
            if call.type != "call_expression":
                continue
            if (setter := SETTERS.get(method_name(call))) is None:
                continue
            base, _ = unroll_chain(call)
            if base.text != b"schema":
                continue
            if (lam := lambda_argument(call, path)) is None:
                continue
            kind, return_type = setter
            name = f"{stem}{kind}{suffix}"
            found.append((lam, name, start_of_line(src, node), return_type))

    edits: list[tuple[int, int, str]] = []
    for lam, name, position, return_type in found:
        if (target := forwarded(lam)) is not None:
            edits.append((lam.start_byte, lam.end_byte, target))
            continue
        if name in taken:
            raise NamingError(f"{location(path, lam)}: name `{name}` is already in use")
        taken.add(name)
        edits.append((position, position, definition(name, return_type, lam)))
        edits.append((lam.start_byte, lam.end_byte, name))
    for start, end, text in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        src = src[:start] + text.encode() + src[end:]
    path.write_bytes(src)
    return len(found)


def main() -> None:
    sources = [
        p for root in (REPO_ROOT / "onnx", REPO_ROOT / "tests") for p in root.rglob("*")
    ]
    taken = {
        m.group()
        for p in sources
        if p.suffix in (".cc", ".h")
        for m in re.finditer(r"\w+", p.read_text())
    }
    total = 0
    try:
        for path in source_files():
            total += name_file(path, taken)
    except NamingError as e:
        sys.exit(str(e))
    print(f"replaced {total} lambdas")


if __name__ == "__main__":
    main()
