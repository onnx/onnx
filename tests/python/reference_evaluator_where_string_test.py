# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import TensorProto, checker, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops import load_op
from onnx.reference.ops.op_where import Where


def _check_selection(condition, x, y, expected, opset):
    arrays = (condition, x, y)
    tensors = [
        numpy_helper.from_array(array, name)
        for name, array in zip(("condition", "X", "Y"), arrays, strict=True)
    ]
    assert tensors[0].data_type == TensorProto.BOOL
    assert tensors[1].data_type == tensors[2].data_type
    for array, tensor in zip(arrays, tensors, strict=True):
        checker.check_tensor(tensor)
        np.testing.assert_array_equal(numpy_helper.to_array(tensor), array)
    graph = helper.make_graph(
        [helper.make_node("Where", ["condition", "X", "Y"], ["output"])],
        "where-selection",
        [
            helper.make_tensor_value_info(
                tensor.name, tensor.data_type, list(tensor.dims)
            )
            for tensor in tensors
        ],
        [
            helper.make_tensor_value_info(
                "output", tensors[1].data_type, list(expected.shape)
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model, full_check=True)
    evaluator = ReferenceEvaluator(model)
    assert load_op("", "Where", opset) is Where
    assert type(evaluator.rt_nodes_[0]) is Where
    originals = [array.copy() for array in arrays]
    output = evaluator.run(
        None, dict(zip(("condition", "X", "Y"), arrays, strict=True))
    )[0]
    assert output.shape == expected.shape
    assert numpy_helper.from_array(output).data_type == tensors[1].data_type
    np.testing.assert_array_equal(output, expected)
    for array, original in zip(arrays, originals, strict=True):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("opset", [9, 16])
@pytest.mark.parametrize(
    ("condition", "x", "y", "expected"),
    [
        pytest.param(
            np.array(False),
            np.array("a"),
            np.array("longer"),
            np.array("longer", dtype=object),
            id="scalar-false",
        ),
        pytest.param(
            np.array([True, False]),
            np.array(["a", "b"]),
            np.array(["longer", "second"]),
            np.array(["a", "second"], dtype=object),
            id="mixed-condition",
        ),
        pytest.param(
            np.array([[False], [True]]),
            np.array(["a", "b"]),
            np.array("extended"),
            np.array([["extended", "extended"], ["a", "b"]], dtype=object),
            id="broadcast-condition",
        ),
        pytest.param(
            np.array(False),
            np.array("甲"),
            np.array("長い文字列"),
            np.array("長い文字列", dtype=object),
            id="unicode-text",
        ),
        pytest.param(
            np.array(False),
            np.array(""),
            np.array("longer"),
            np.array("longer", dtype=object),
            id="empty-string-x",
        ),
        pytest.param(
            np.array([False, True]),
            np.array(["a", "b"]),
            np.array(["longer", "second"], dtype=object),
            np.array(["longer", "b"], dtype=object),
            id="object-y",
        ),
    ],
)
def test_selects_complete_strings(condition, x, y, expected, opset):
    _check_selection(condition, x, y, expected, opset)


@pytest.mark.parametrize("opset", [9, 16])
@pytest.mark.parametrize(
    ("condition", "x", "y", "expected"),
    [
        pytest.param(
            np.array(True),
            np.array("longer"),
            np.array("a"),
            np.array("longer", dtype=object),
            id="scalar-true",
        ),
        pytest.param(
            np.array(False),
            np.array("longer"),
            np.array("a"),
            np.array("a", dtype=object),
            id="shorter-y",
        ),
        pytest.param(
            np.array([False, True]),
            np.array(["a", "b"], dtype=object),
            np.array(["longer", "second"], dtype=object),
            np.array(["longer", "b"], dtype=object),
            id="canonical-object-strings",
        ),
        pytest.param(
            np.array([True, False]),
            np.array(["longer", "shorts"]),
            np.array(["second", "second"]),
            np.array(["longer", "second"], dtype=object),
            id="same-width-strings",
        ),
        pytest.param(
            np.empty((0,), dtype=bool),
            np.array("a"),
            np.array("longer"),
            np.empty((0,), dtype=object),
            id="empty-vector-condition",
        ),
        pytest.param(
            np.empty((0, 3), dtype=bool),
            np.array([["a", "b", "c"]]),
            np.array("longer"),
            np.empty((0, 3), dtype=object),
            id="empty-broadcast-condition",
        ),
        pytest.param(
            np.array([[True, False], [True, True]]),
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[9, 8], [7, 6]], dtype=np.float32),
            np.array([[1, 8], [3, 4]], dtype=np.float32),
            id="existing-float-example",
        ),
        pytest.param(
            np.array([[True, False], [True, True]]),
            np.array([[1, 2], [3, 4]], dtype=np.int64),
            np.array([[9, 8], [7, 6]], dtype=np.int64),
            np.array([[1, 8], [3, 4]], dtype=np.int64),
            id="existing-int64-example",
        ),
        pytest.param(
            np.array([[True], [False]]),
            np.array([True, False]),
            np.array([False, True]),
            np.array([[True, False], [False, True]]),
            id="boolean-data-broadcast",
        ),
    ],
)
def test_supported_conditions_and_dtypes(condition, x, y, expected, opset):
    _check_selection(condition, x, y, expected, opset)
