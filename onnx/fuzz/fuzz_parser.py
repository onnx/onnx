# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import re
import sys
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING

import atheris

with atheris.instrument_imports():
    from onnx import parser, printer

if TYPE_CHECKING:
    from collections.abc import Callable

    from onnx import NodeProto


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


def _check_float_literal(
    body: str, expected: float, special: bool, get: Callable[[NodeProto], float]
) -> None:
    # Decimal literals that overflow the target type must be rejected.
    overflow = math.isinf(expected) and not special
    try:
        node = parser.parse_node(body)
    except parser.ParseError:
        assert overflow, (body, "Valid floating-point literal was rejected")
    else:
        assert not overflow, (body, "Overflowing floating-point literal was accepted")
        _check_float(get(node), expected)


def _check_integer(text: str) -> None:
    # Normalize first so long literals do not hit Python's int string limit.
    digits = text.removeprefix("-").lstrip("0") or "0"
    value = None
    if len(digits) <= 20:
        value = int(digits)
        if text.startswith("-"):
            value = -value
    for kind in ("int", "int64", "uint64"):
        lower, upper = (0, 2**64 - 1) if kind == "uint64" else (-(2**63), 2**63 - 1)
        valid = value is not None and lower <= value <= upper
        # Unsigned literals must not carry a sign, even "-0".
        if kind == "uint64" and text.startswith("-"):
            valid = False
        body = (
            f"value_int: int={text}" if kind == "int" else f"value={kind}[1] {{{text}}}"
        )
        try:
            node = parser.parse_node(f"Y = Constant<{body}>()")
        except parser.ParseError:
            assert not valid, (kind, text, "Valid integer literal was rejected")
        else:
            assert valid, (kind, text, "Out-of-range integer literal was accepted")
            attribute = node.attribute[0]
            if kind == "int":
                actual = attribute.i
            elif kind == "int64":
                actual = attribute.t.int64_data[0]
            else:
                actual = attribute.t.uint64_data[0]
            assert actual == value, (kind, text, actual, value)


def _check_round_trip(parse, proto) -> None:
    """Printing a parsed proto must give text that parses to the same text."""
    printed = printer.to_text(proto)
    try:
        reparsed = parse(printed)
    except parser.ParseError as error:
        error.add_note(f"Parser entry point: {parse.__name__}")
        error.add_note(printed)
        raise AssertionError("Printed text does not parse") from error
    assert printer.to_text(reparsed) == printed, (
        parse.__name__,
        printed,
        "Printed text changed after parsing it back",
    )


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
            proto = parse(text)
        except parser.ParseError:
            # Malformed ONNX text is expected on the raw-input path.
            continue
        except Exception as error:
            error.add_note(f"Parser entry point: {parse.__name__}")
            raise
        _check_round_trip(parse, proto)
    if re.fullmatch(r"-?[0-9]+", text):
        _check_integer(text)
    # Decimal literals parse unless they overflow; underflow rounds to zero or a
    # subnormal. Special values (inf, infinity, nan) always parse.
    special = text.lower().removeprefix("-") in {"inf", "infinity", "nan"}
    if special or re.fullmatch(r"-?[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?", text):
        expected = float(text)
        _check_float_literal(
            f"Y = Constant<value_float: float={text}>()",
            _float32(text, expected),
            special,
            lambda node: node.attribute[0].f,
        )
        _check_float_literal(
            f"Y = Constant<value=double[1] {{{text}}}>()",
            expected,
            special,
            lambda node: node.attribute[0].t.double_data[0],
        )
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
