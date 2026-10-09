# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import locale

import numpy as np
import pytest

from onnx import TensorProto, checker, defs, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops import _op_list
from onnx.reference.ops.op_string_normalizer import StringNormalizer


@pytest.fixture(scope="module", autouse=True)
def c_locale():
    previous = locale.setlocale(locale.LC_ALL)
    try:
        locale.setlocale(locale.LC_ALL, "C")
    except locale.Error:
        pytest.skip("The C locale is unavailable")
    try:
        yield
    finally:
        locale.setlocale(locale.LC_ALL, previous)


CASES = [
    pytest.param(
        ["", "Monday", ""],
        ["", "Monday", ""],
        {"is_case_sensitive": 1},
        id="default_none_preserves_empty_vector",
    ),
    pytest.param(
        [["", "Monday"]],
        [["", "MONDAY"]],
        {"case_change_action": "UPPER"},
        id="upper_preserves_empty_row",
    ),
    pytest.param(
        ["", "MonDAY", ""],
        ["", "monday", ""],
        {"case_change_action": "LOWER"},
        id="lower_preserves_empty_vector",
    ),
    pytest.param(
        [["", ""]],
        [["", ""]],
        {"case_change_action": "NONE"},
        id="none_preserves_multiple_empty_tokens",
    ),
    pytest.param(
        ["", "monday", "Tuesday"],
        ["", "Tuesday"],
        {"stopwords": ["monday"], "is_case_sensitive": 1},
        id="empty_token_survives_other_stopword",
    ),
    pytest.param(
        [["", "Monday"]],
        [["", "monday"]],
        {
            "stopwords": ["stop"],
            "case_change_action": "LOWER",
            "is_case_sensitive": 1,
        },
        id="empty_token_survives_nonmatching_stopword",
    ),
    pytest.param(
        ["", "Monday"],
        ["Monday"],
        {"stopwords": [""], "is_case_sensitive": 1},
        id="explicit_empty_stopword_is_removed",
    ),
    pytest.param(
        ["monday", "monday"],
        [""],
        {
            "stopwords": ["monday"],
            "case_change_action": "UPPER",
            "is_case_sensitive": 1,
        },
        id="all_stopwords_vector_keeps_empty_fallback",
    ),
    pytest.param(
        [["monday", "Monday"]],
        [[""]],
        {"stopwords": ["monday"], "case_change_action": "UPPER"},
        id="all_stopwords_row_keeps_empty_fallback",
    ),
    pytest.param(
        ["Monday", "Tuesday"],
        ["monday", "tuesday"],
        {
            "stopwords": ["monday"],
            "case_change_action": "LOWER",
            "is_case_sensitive": 1,
        },
        id="case_sensitive_filter_precedes_lowercase",
    ),
    pytest.param(
        ["monday", "tuesday"],
        ["monday", "tuesday"],
        {"is_case_sensitive": 1},
        id="ordinary_noop",
    ),
]


@pytest.mark.parametrize(("values", "expected_values", "attributes"), CASES)
def test_string_normalizer_empty_tokens(values, expected_values, attributes):
    schema = defs.get_schema("StringNormalizer", 10, "")
    assert schema.since_version == 10
    assert schema.inputs[0].type_str == "tensor(string)"
    assert attributes.get("case_change_action", "NONE") in {"NONE", "LOWER", "UPPER"}

    original = np.array(values, dtype=object)
    assert original.ndim == 1 or (original.ndim == 2 and original.shape[0] == 1)
    assert original.shape[-1] > 0
    tensor = numpy_helper.from_array(original, name="X")
    tensor = TensorProto.FromString(tensor.SerializeToString())
    checker.check_tensor(tensor)
    assert tensor.data_type == TensorProto.STRING
    x = numpy_helper.to_array(tensor)
    np.testing.assert_array_equal(x, original)
    expected = np.array(expected_values, dtype=object)

    node = helper.make_node("StringNormalizer", ["X"], ["Y"], locale="C", **attributes)
    graph = helper.make_graph(
        [node],
        "string_normalizer_empty_tokens",
        [helper.make_tensor_value_info("X", TensorProto.STRING, x.shape)],
        [helper.make_tensor_value_info("Y", TensorProto.STRING, expected.shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model = type(model).FromString(model.SerializeToString())
    checker.check_model(model, full_check=True)
    assert _op_list.load_op("", "StringNormalizer", 10) is StringNormalizer

    evaluator = ReferenceEvaluator(model)
    assert len(evaluator.rt_nodes_) == 1
    runtime_node = evaluator.rt_nodes_[0]
    assert type(runtime_node) is StringNormalizer
    assert runtime_node._run.__func__ is StringNormalizer._run
    (actual,) = evaluator.run(None, {"X": x})
    assert actual.dtype.kind in {"O", "U"}
    np.testing.assert_array_equal(actual, expected)
