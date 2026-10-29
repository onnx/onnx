# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import sys

import atheris

with atheris.instrument_imports():
    from onnx import parser


def TestOneInput(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return
    for parse in (
        parser.parse_model,
        parser.parse_graph,
        parser.parse_function,
        parser.parse_node,
    ):
        try:
            parse(text)
        except parser.ParseError:
            pass
        except Exception as error:
            error.add_note(f"Parser entry point: {parse.__name__}")
            raise
    # Decimal literals are valid floating-point attributes even on range errors.
    # Do not suppress ParseError here: rejecting such a literal is also a bug.
    if re.fullmatch(r"-?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?", text):
        parser.parse_node(f"Y = Constant<value_float: float={text}>()")
    # Escape only quotes and backslashes, as required by the ONNX text format.
    quoted = text.replace("\\", "\\\\").replace('"', '\\"')
    node = parser.parse_node(f'Y = Constant<value_string="{quoted}">()')
    assert node.attribute[0].s == data, "String literal was not preserved"


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
