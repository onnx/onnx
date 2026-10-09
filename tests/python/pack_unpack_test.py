# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import checker, defs, helper, parser
from onnx.reference import ReferenceEvaluator


def evaluator(
    op: str, bits: int, function: bool = False, shape: tuple[int, ...] = (1,)
) -> ReferenceEvaluator:
    dimensions = ", ".join(map(str, shape))
    output_dimensions = ", ".join([*map(str, shape[:-1]), "N"])
    signature = f"uint8[{dimensions}] X"
    inputs = "X"
    if op == "Unpack":
        signature += ", int64 count"
        inputs += ", count"
    model = parser.parse_model(
        f"""
        <ir_version: 14, opset_import: ["" : 29]>
        g ({signature}) => (uint8[{output_dimensions}] Y) {{
            Y = {op}<bits = {bits}>({inputs})
        }}
        """
    )
    if function:
        body = defs.get_schema(op).function_body
        body.domain = "test.packing"
        model.functions.append(body)
        model.graph.node[0].domain = body.domain
        model.opset_import.append(helper.make_opsetid(body.domain, 1))
    checker.check_model(model, full_check=function)
    return ReferenceEvaluator(model)


def oracle(x: np.ndarray, bits: int) -> np.ndarray:
    n = x.shape[-1]
    packed = np.empty((*x.shape[:-1], (n * bits + 7) // 8), dtype=np.uint8)
    for index in np.ndindex(x.shape[:-1]):
        word = sum(int(code) << (i * bits) for i, code in enumerate(x[index]))
        packed[index] = np.frombuffer(
            word.to_bytes(packed.shape[-1], "little"), dtype=np.uint8
        )
    return packed


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("n", [0, 1, 2, 7, 8, 9, 17])
@pytest.mark.parametrize("leading", [(), (2,), (2, 3), (0,), (2, 0)])
def test_pack_unpack(bits: int, n: int, leading: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1 << bits, size=(*leading, n), dtype=np.uint8)
    expected = oracle(x, bits)
    for function in (False, True):
        packed = evaluator("Pack", bits, function, x.shape).run(None, {"X": x})[0]
        np.testing.assert_array_equal(packed, expected)
        unpacked = evaluator("Unpack", bits, function, packed.shape).run(
            None, {"X": packed, "count": np.array(n, dtype=np.int64)}
        )[0]
        np.testing.assert_array_equal(unpacked, x)


@pytest.mark.parametrize("function", [False, True])
def test_unpack_ignores_padding(function: bool) -> None:
    x = np.array([[0xFF]], dtype=np.uint8)
    y = evaluator("Unpack", 3, function, x.shape).run(
        None, {"X": x, "count": np.array(2, dtype=np.int64)}
    )[0]
    np.testing.assert_array_equal(y, [[7, 7]])


@pytest.mark.parametrize("bits", [0, 9, -1])
@pytest.mark.parametrize("op", ["Pack", "Unpack"])
def test_invalid_bits(op: str, bits: int) -> None:
    feeds = {"X": np.array([0], dtype=np.uint8)}
    if op == "Unpack":
        feeds["count"] = np.array(1, dtype=np.int64)
    with pytest.raises(ValueError, match="bits must be"):
        evaluator(op, bits).run(None, feeds)


@pytest.mark.parametrize("op", ["Pack", "Unpack"])
def test_scalar_input(op: str) -> None:
    feeds = {"X": np.array(0, dtype=np.uint8)}
    if op == "Unpack":
        feeds["count"] = np.array(1, dtype=np.int64)
    with pytest.raises(ValueError, match="rank at least 1"):
        evaluator(op, 3).run(None, feeds)


def test_pack_code_range() -> None:
    with pytest.raises(ValueError, match="representable"):
        evaluator("Pack", 3).run(None, {"X": np.array([8], dtype=np.uint8)})


@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("function", [False, True])
def test_noncontiguous_input(bits: int, function: bool) -> None:
    x = (np.arange(48) % (1 << bits)).astype(np.uint8).reshape(4, 12)[:, ::2]
    packed = evaluator("Pack", bits, function, x.shape).run(None, {"X": x})[0]
    np.testing.assert_array_equal(packed, oracle(x, bits))
    padded = np.zeros((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.uint8)
    padded[:, ::2] = packed
    sliced = padded[:, ::2]
    unpacked = evaluator("Unpack", bits, function, sliced.shape).run(
        None, {"X": sliced, "count": np.array(x.shape[-1], dtype=np.int64)}
    )[0]
    np.testing.assert_array_equal(unpacked, x)


@pytest.mark.parametrize(
    "count", [np.array(-1, dtype=np.int64), np.array([2], dtype=np.int64)]
)
def test_unpack_invalid_count(count: np.ndarray) -> None:
    with pytest.raises(ValueError, match="nonnegative scalar"):
        evaluator("Unpack", 3).run(
            None, {"X": np.array([0], dtype=np.uint8), "count": count}
        )


@pytest.mark.parametrize("count", [0, 3])
def test_unpack_invalid_length(count: int) -> None:
    with pytest.raises(ValueError, match="Packed last dimension"):
        evaluator("Unpack", 3).run(
            None,
            {
                "X": np.array([0], dtype=np.uint8),
                "count": np.array(count, dtype=np.int64),
            },
        )
