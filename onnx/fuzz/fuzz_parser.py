# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import re
import sys
from decimal import Decimal
from fractions import Fraction

import atheris

with atheris.instrument_imports():
    from onnx import parser


def _float32(text: str, value: float) -> float:
    if not math.isfinite(value) or value == 0:
        return value
    if abs(value) >= 2**128:
        return math.copysign(math.inf, value)
    if abs(value) < 2**-150:
        return math.copysign(0.0, value)
    # Round the decimal directly to binary32, avoiding double rounding.
    exact = abs(Fraction(Decimal(text)))
    numerator, denominator = exact.as_integer_ratio()
    exponent = numerator.bit_length() - denominator.bit_length()
    if exact < Fraction(2) ** exponent:
        exponent -= 1
    quantum = max(exponent - 23, -149)
    scale = Fraction(2) ** quantum
    rounded = math.ldexp(round(exact / scale), quantum)
    return math.copysign(math.inf if rounded >= 2**128 else rounded, value)


def _check_float(actual: float, expected: float) -> None:
    if math.isnan(expected):
        assert math.isnan(actual), (actual, expected)
    else:
        # hex() compares exact values and distinguishes positive/negative zero.
        assert actual.hex() == expected.hex(), (actual, expected)


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
            # Malformed ONNX text is expected on the raw-input path.
            pass
        except Exception as error:
            error.add_note(f"Parser entry point: {parse.__name__}")
            raise
    # Decimal literals are valid floating-point attributes even on range errors.
    # Do not suppress ParseError here: rejecting such a literal is also a bug.
    if re.fullmatch(
        r"-?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?", text
    ) or text.lower().removeprefix("-") in {"inf", "infinity", "nan"}:
        expected = float(text)
        node = parser.parse_node(f"Y = Constant<value_float: float={text}>()")
        _check_float(node.attribute[0].f, _float32(text, expected))
        node = parser.parse_node(f"Y = Constant<value=double[1] {{{text}}}>()")
        _check_float(node.attribute[0].t.double_data[0], expected)
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
