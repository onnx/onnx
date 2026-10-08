# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import TensorProto, checker, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops import load_op
from onnx.reference.ops.op_string_split import StringSplit


def _check_split(x, delimiter, maxsplit, expected, num_tokens):
    tensor = numpy_helper.from_array(x, "X")
    assert tensor.data_type == TensorProto.STRING
    checker.check_tensor(tensor)
    np.testing.assert_array_equal(numpy_helper.to_array(tensor), x)
    attributes = {}
    if delimiter is not None:
        attributes["delimiter"] = delimiter
    if maxsplit is not None:
        attributes["maxsplit"] = maxsplit
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("StringSplit", ["X"], ["Y", "Z"], **attributes)],
            "string-split",
            [helper.make_tensor_value_info("X", TensorProto.STRING, list(x.shape))],
            [
                helper.make_tensor_value_info(
                    "Y", TensorProto.STRING, list(expected.shape)
                ),
                helper.make_tensor_value_info(
                    "Z", TensorProto.INT64, list(num_tokens.shape)
                ),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 20)],
    )
    checker.check_model(model, full_check=True)
    evaluator = ReferenceEvaluator(model)
    assert load_op("", "StringSplit", 20) is StringSplit
    assert type(evaluator.rt_nodes_[0]) is StringSplit
    original = x.copy()
    output, counts = evaluator.run(None, {"X": x})
    assert output.shape == expected.shape
    assert counts.shape == x.shape
    assert counts.dtype == np.int64
    assert numpy_helper.from_array(output).data_type == TensorProto.STRING
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(counts, num_tokens)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("dtype", [np.str_, object], ids=["unicode", "object"])
@pytest.mark.parametrize(
    ("value", "delimiter", "maxsplit", "tokens", "num_tokens"),
    [
        pytest.param("a,b", ",", None, ["a", "b"], 2, id="comma"),
        pytest.param("", None, None, [], 0, id="empty-default"),
        pytest.param("", "", None, [], 0, id="empty-whitespace-delimiter"),
        pytest.param("", ",", None, [""], 1, id="empty-comma"),
        pytest.param(
            " \talpha  beta \n",
            None,
            None,
            ["alpha", "beta"],
            2,
            id="whitespace",
        ),
        pytest.param(
            "a||b||||", "||", None, ["a", "b", "", ""], 4, id="substring-delimiter"
        ),
        pytest.param("甲☃乙☃", "☃", None, ["甲", "乙", ""], 3, id="unicode-delimiter"),
        pytest.param("a,b,c", ",", 1, ["a", "b,c"], 2, id="maxsplit-one"),
        pytest.param("a,b", ",", 0, ["a,b"], 1, id="maxsplit-zero"),
    ],
)
def test_scalar_string_split(value, delimiter, maxsplit, tokens, num_tokens, dtype):
    _check_split(
        np.array(value, dtype=dtype),
        delimiter,
        maxsplit,
        np.array(tokens, dtype=object),
        np.array(num_tokens, dtype=np.int64),
    )


@pytest.mark.parametrize("dtype", [np.str_, object], ids=["unicode", "object"])
@pytest.mark.parametrize(
    ("x", "delimiter", "maxsplit", "expected", "num_tokens"),
    [
        pytest.param(
            np.array(["abc.com", "def.net"], dtype=object),
            ".",
            None,
            np.array([["abc", "com"], ["def", "net"]], dtype=object),
            np.array([2, 2], dtype=np.int64),
            id="existing-basic",
        ),
        pytest.param(
            np.array(["1,", "4,6", ""], dtype=object),
            ",",
            None,
            np.array([["1", ""], ["4", "6"], ["", ""]], dtype=object),
            np.array([2, 2, 1], dtype=np.int64),
            id="empty-token-vs-padding",
        ),
        pytest.param(
            np.array(
                [["hello world", "def.net"], ["o n n x", "the quick brown fox"]],
                dtype=object,
            ),
            None,
            2,
            np.array(
                [
                    [["hello", "world", ""], ["def.net", "", ""]],
                    [["o", "n", "n x"], ["the", "quick", "brown fox"]],
                ],
                dtype=object,
            ),
            np.array([[2, 1], [3, 3]], dtype=np.int64),
            id="existing-matrix-maxsplit",
        ),
        pytest.param(
            np.array(["", " \t", "a b"], dtype=object),
            None,
            None,
            np.array([["", ""], ["", ""], ["a", "b"]], dtype=object),
            np.array([0, 0, 2], dtype=np.int64),
            id="zero-tokens-vs-padding",
        ),
        pytest.param(
            np.array([" a  b ", "\t"], dtype=object),
            "",
            None,
            np.array([["a", "b"], ["", ""]], dtype=object),
            np.array([2, 0], dtype=np.int64),
            id="empty-delimiter",
        ),
        pytest.param(
            np.empty((0,), dtype=object),
            None,
            None,
            np.empty((0, 0), dtype=object),
            np.empty((0,), dtype=np.int64),
            id="existing-empty-vector",
        ),
        pytest.param(
            np.empty((0, 3), dtype=object),
            ",",
            1,
            np.empty((0, 3, 0), dtype=object),
            np.empty((0, 3), dtype=np.int64),
            id="empty-first-dimension",
        ),
        pytest.param(
            np.empty((2, 0), dtype=object),
            "",
            None,
            np.empty((2, 0, 0), dtype=object),
            np.empty((2, 0), dtype=np.int64),
            id="empty-last-dimension",
        ),
        pytest.param(
            np.array([["", " \t"], ["\n", ""]], dtype=object),
            None,
            None,
            np.empty((2, 2, 0), dtype=object),
            np.zeros((2, 2), dtype=np.int64),
            id="matrix-with-no-tokens",
        ),
    ],
)
def test_string_split_controls(x, delimiter, maxsplit, expected, num_tokens, dtype):
    _check_split(x.astype(dtype), delimiter, maxsplit, expected, num_tokens)
