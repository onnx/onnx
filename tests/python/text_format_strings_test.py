# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import onnx
from onnx import helper, parser, printer


@pytest.mark.parametrize(
    ("text", "context"),
    [
        ("", ""),
        (" \t\n\v\f\r", " \t"),
        ("\n@", ""),
        ("X = @", "X = @"),
        ("# comment\nX = @\n", "X = @"),
        ("X = Op(\n \t", "X = Op("),
        ("X = Op(\r\n \t", "X = Op(\r"),
        ("# comment without newline", "# comment without newline"),
    ],
)
def test_error_context(text: str, context: str) -> None:
    with pytest.raises(onnx.parser.ParseError) as exc:
        onnx.parser.parse_node(text)
    message = exc.value.args[0]
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    assert f"Error context: {context}\n" in message


@pytest.mark.parametrize("text", ['"', '"unterminated', '"tail\\', '"escaped\\"'])
def test_incomplete_string_literal(text: str) -> None:
    with pytest.raises(onnx.parser.ParseError, match="Incomplete string literal"):
        onnx.parser.parse_node(text)


@pytest.mark.parametrize("whitespace", [" ", "\t", "\n", "\v", "\f", "\r", "\r\n"])
def test_whitespace_and_comments(whitespace: str) -> None:
    text = whitespace.join(["Y", "=", "Identity", "(", "X", ")"])
    node = onnx.parser.parse_node(
        f"{whitespace}# leading comment\n# another comment\n{text}# trailing comment"
    )
    assert node.op_type == "Identity"
    assert list(node.input) == ["X"]
    assert list(node.output) == ["Y"]


@pytest.mark.parametrize(
    "name",
    [
        "",
        "plain",
        "two words",
        '"',
        "\\",
        '\\"',
        "tail\\",
        "a\x00b",
        "前綴",
        "a" * 4096 + '"tail\\',
    ],
)
def test_quoted_identifier_escaping(name: str) -> None:
    node = helper.make_node("Identity", [name], ["Y"])
    text = printer.to_text(node)
    restored = parser.parse_node(text)
    assert list(restored.input) == [name]
    assert printer.to_text(restored) == text
