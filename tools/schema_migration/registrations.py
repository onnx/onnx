# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="union-attr,assignment,arg-type,return-value,index"
"""Operator registrations and schema generators in 'onnx/defs'.

Shared by the migration scripts. A registration is a call of one of the
'MACROS' whose third argument is a chain of builder calls on 'OpSchema()' or on a
generated schema function 'generated::<domain>::<Name>_ver<N>()'.
Schema generators are functions returning 'std::function<void(OpSchema&)>';
registrations use them via '.FillUsing(Generator(...))'.

Requires 'tree-sitter' and 'tree-sitter-cpp'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFS_DIR = REPO_ROOT / "onnx" / "defs"
PARSER = Parser(Language(tree_sitter_cpp.language()))

MACROS = {
    b"ONNX_OPERATOR_SET_SCHEMA": "ai.onnx",
    b"ONNX_ML_OPERATOR_SET_SCHEMA": "ai.onnx.ml",
    b"ONNX_PREVIEW_OPERATOR_SET_SCHEMA": "ai.onnx.preview",
    b"ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA": "ai.onnx.preview.training",
}


class RegistrationError(Exception):
    pass


def walk(node: Node):
    yield node
    for child in node.children:
        yield from walk(child)


def source_files() -> list[Path]:
    return sorted(DEFS_DIR.glob("*/*.cc"))


def location(path: Path, node: Node) -> str:
    return f"{path.relative_to(REPO_ROOT)}:{node.start_point[0] + 1}"


def method_name(call: Node) -> bytes | None:
    """Return `B` of a method call `a.B(...)`, else None."""
    function = call.child_by_field_name("function")
    if function is None or function.type != "field_expression":
        return None
    return function.child_by_field_name("field").text


def unroll_chain(expr: Node) -> tuple[Node, list[Node]]:
    """Split `base.A(x).B(y)` into `base` and the method calls `A(x)`, `B(y)`."""
    calls: list[Node] = []
    node = expr
    while node.type == "call_expression" and method_name(node) is not None:
        calls.append(node)
        node = node.child_by_field_name("function").child_by_field_name("argument")
    return node, calls[::-1]


def arguments(call: Node) -> list[Node]:
    return [
        c
        for c in call.child_by_field_name("arguments").named_children
        if c.type != "comment"
    ]


@dataclass
class Generator:
    name: str
    path: Path
    node: Node
    # Statements `schema.X(...);` of the generator body, in order.
    statements: list[Node] = field(default_factory=list)
    # The generator call `G(...)` if the body is `return G(...);`.
    forward: Node | None = None


def is_generator(node: Node) -> bool:
    if node.type != "function_definition":
        return False
    head = node.text[: node.child_by_field_name("body").start_byte - node.start_byte]
    return b"std::function<void(OpSchema&)>" in head


def generator_name(node: Node) -> str:
    declarator = node.child_by_field_name("declarator")
    while declarator.type != "function_declarator":
        declarator = declarator.child_by_field_name("declarator")
    return declarator.child_by_field_name("declarator").text.decode()


def schema_statements(body: Node) -> list[Node]:
    """Return the statements `schema.X(...)...;` in `body`."""
    out = []
    for node in walk(body):
        if node.type != "expression_statement" or not node.named_children:
            continue
        base, calls = unroll_chain(node.named_children[0])
        if base.type == "identifier" and base.text == b"schema" and calls:
            out.append(node)
    return out


def forwarded_call(body: Node) -> Node | None:
    """Return `G(...)` if `body` is `{ return G(...); }`."""
    statements = [c for c in body.named_children if c.type != "comment"]
    if len(statements) != 1 or statements[0].type != "return_statement":
        return None
    (value,) = statements[0].named_children
    return value if value.type == "call_expression" else None


def generators() -> dict[str, Generator]:
    out: dict[str, Generator] = {}
    for path in source_files() + sorted(DEFS_DIR.glob("*/*.h")):
        root = PARSER.parse(path.read_bytes()).root_node
        for node in walk(root):
            if is_generator(node):
                name = generator_name(node)
                if name in out:
                    raise RegistrationError(
                        f"{location(path, node)}: duplicate generator `{name}`"
                    )
                body = node.child_by_field_name("body")
                out[name] = Generator(
                    name, path, node, schema_statements(body), forwarded_call(body)
                )
    return out


@dataclass
class Registration:
    domain: str
    name: str
    version: int
    path: Path
    # The macro call and the builder chain passed to it.
    node: Node
    base: Node
    calls: list[Node]


def registrations(path: Path, root: Node) -> list[Registration]:
    out = []
    for node in walk(root):
        if node.type != "call_expression":
            continue
        domain = MACROS.get(node.child_by_field_name("function").text)
        if domain is None:
            continue
        args = arguments(node)
        if len(args) != 3:  # noqa: PLR2004
            raise RegistrationError(f"{location(path, node)}: unexpected arguments")
        name, version, expr = args
        base, calls = unroll_chain(expr)
        if base.type != "call_expression" or not (
            base.text == b"OpSchema()" or base.text.startswith(b"generated::")
        ):
            raise RegistrationError(f"{location(path, node)}: not an OpSchema chain")
        out.append(
            Registration(
                domain,
                name.text.decode(),
                int(version.text),
                path,
                node,
                base,
                calls,
            )
        )
    return out
