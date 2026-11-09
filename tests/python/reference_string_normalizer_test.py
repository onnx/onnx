# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


@pytest.mark.parametrize("opset", [10, 29])
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dtype", [object, str])
@pytest.mark.parametrize("action", ["NONE", "LOWER", "UPPER"])
@pytest.mark.parametrize("sensitive", [False, True])
@pytest.mark.parametrize(
    "values,stops,kept_sensitive,kept_insensitive",
    [
        (["Monday", "monday", "Keep"], ["monday"], ["Monday", "Keep"], ["Keep"]),
        (["the cat", "the", "Keep"], ["the"], ["the cat", "Keep"], ["the cat", "Keep"]),
        (
            ["new york", "New York", "new", "york"],
            ["new york"],
            ["New York", "new", "york"],
            ["new", "york"],
        ),
        (
            ["  the ", "the\tcat", "the\ncat", "the"],
            ["the"],
            ["  the ", "the\tcat", "the\ncat"],
            ["  the ", "the\tcat", "the\ncat"],
        ),
        (["", "Keep", ""], ["other"], ["", "Keep", ""], ["", "Keep", ""]),
        (["", "Keep", ""], [""], ["Keep"], ["Keep"]),
        (["café", "cafe", "CAFÉ"], ["café"], ["cafe", "CAFÉ"], ["cafe"]),
        (
            ["café", "naïve", "ß", "İ"],
            None,
            ["café", "naïve", "ß", "İ"],
            ["café", "naïve", "ß", "İ"],
        ),
        (["the", "the"], ["the", "the"], [], []),
        (["ß", "İ"], None, ["ß", "İ"], ["ß", "İ"]),
    ],
)
def test_string_normalizer_elements(
    opset,
    matrix,
    dtype,
    action,
    sensitive,
    values,
    stops,
    kept_sensitive,
    kept_insensitive,
):
    x = np.array(values, dtype=dtype)
    if matrix:
        x = x.reshape(1, -1)
    original = x.copy()
    attributes = {
        "case_change_action": action,
        "is_case_sensitive": int(sensitive),
        "locale": "C",
    }
    if stops is not None:
        attributes["stopwords"] = stops
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("StringNormalizer", ["x"], ["y"], **attributes)],
            "string_normalizer_elements",
            [helper.make_tensor_value_info("x", TensorProto.STRING, list(x.shape))],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.STRING, [1, None] if matrix else [None]
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    kept = kept_sensitive if sensitive else kept_insensitive
    if action == "LOWER":
        kept = [value.lower() for value in kept]
    elif action == "UPPER":
        kept = [value.upper() for value in kept]
    expected = np.array(kept or [""], dtype=object)
    if matrix:
        expected = expected.reshape(1, -1)
    actual = ReferenceEvaluator(model).run(None, {"x": x})[0]
    assert actual.shape == expected.shape
    assert actual.dtype.kind in {"O", "U"}
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("matrix", [False, True])
def test_string_normalizer_default_attributes(matrix):
    x = np.array(["Monday", "monday", "Keep", ""], dtype=object)
    expected = np.array(["Keep", ""], dtype=object)
    if matrix:
        x = x.reshape(1, -1)
        expected = expected.reshape(1, -1)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "StringNormalizer", ["x"], ["y"], stopwords=["monday"], locale="C"
                )
            ],
            "string_normalizer_defaults",
            [helper.make_tensor_value_info("x", TensorProto.STRING, list(x.shape))],
            [
                helper.make_tensor_value_info(
                    "y", TensorProto.STRING, [1, None] if matrix else [None]
                )
            ],
        ),
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.checker.check_model(model, full_check=True)
    np.testing.assert_array_equal(
        ReferenceEvaluator(model).run(None, {"x": x})[0], expected
    )
