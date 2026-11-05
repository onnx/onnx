# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import onnx
    from onnx import parser, printer


def TestOneInput(data):
    try:
        model = onnx.load_model_from_string(data)
        _ = len(model.graph.node)
        _ = len(model.graph.input)
        _ = len(model.graph.output)
        onnx.checker.check_model(model)
        text = printer.to_text(model)
    except Exception:
        return
    # The text form of a valid model must parse back to the same text.
    try:
        reparsed = parser.parse_model(text)
    except parser.ParseError as error:
        error.add_note(text)
        raise AssertionError("Printed text of a valid model does not parse") from error
    assert printer.to_text(reparsed) == text, (
        text,
        "Printed text changed after parsing it back",
    )


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
