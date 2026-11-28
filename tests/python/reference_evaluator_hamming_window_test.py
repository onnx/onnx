# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import onnx
from onnx import TensorProto, helper, version_converter
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRunExpand


class HammingWindow(OpRunExpand):
    op_domain = ""


def make_window_model(opset, periodic, dtype, size=None):
    attributes = {"periodic": periodic, "output_datatype": dtype}
    attributes = {
        name: value for name, value in attributes.items() if value is not None
    }
    node = helper.make_node("HammingWindow", ["size"], ["y"], **attributes)
    return helper.make_model(
        helper.make_graph(
            [node],
            "hamming",
            [helper.make_tensor_value_info("size", TensorProto.INT64, [])],
            [helper.make_tensor_value_info("y", dtype or TensorProto.FLOAT, [size])],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )


@pytest.mark.parametrize("opset", [17, 28, 29])
@pytest.mark.parametrize("periodic", [0, 1])
@pytest.mark.parametrize(
    "dtype", [TensorProto.FLOAT16, TensorProto.FLOAT, TensorProto.DOUBLE]
)
@pytest.mark.parametrize("size", [2, 11])
@pytest.mark.parametrize("expand", [False, True])
def test_hamming_coefficients_by_opset(opset, periodic, dtype, size, expand):
    model = make_window_model(opset, periodic, dtype, size)
    onnx.checker.check_model(model, full_check=True)
    runtime = ReferenceEvaluator(model, new_ops=[HammingWindow] if expand else None)
    actual = runtime.run(None, {"size": np.array(size, dtype=np.int64)})[0]
    alpha, beta = (0.54, 0.46) if opset >= 29 else (25 / 46, 21 / 46)
    denominator = size if periodic else size - 1
    expected = alpha - beta * np.cos(2 * np.pi * np.arange(size) / denominator)
    output_dtype = helper.tensor_dtype_to_np_dtype(dtype)
    assert actual.dtype == output_dtype
    assert_allclose(
        actual,
        expected.astype(output_dtype),
        rtol=2e-3 if dtype == TensorProto.FLOAT16 else 1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("source,target", [(28, 29), (29, 28), (17, 29), (29, 17)])
@pytest.mark.parametrize("periodic", [0, 1, None])
@pytest.mark.parametrize(
    "dtype", [None, TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.INT32]
)
def test_hamming_conversion_keeps_source_coefficients(source, target, periodic, dtype):
    model = make_window_model(source, periodic, dtype)
    converted = version_converter.convert_version(model, target)
    onnx.checker.check_model(converted, full_check=True)
    assert converted.opset_import[0].version == target
    assert all(node.op_type != "HammingWindow" for node in converted.graph.node)
    before = ReferenceEvaluator(model, new_ops=[HammingWindow])
    after = ReferenceEvaluator(converted)
    for size in (2, 7, 16):
        feeds = {"size": np.array(size, dtype=np.int64)}
        expected = before.run(None, feeds)[0]
        actual = after.run(None, feeds)[0]
        # Both use the source schema function, including FLOAT intermediates
        # before the requested output cast. Converting must not round through
        # the other version's window values.
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("source,target", [(28, 29), (29, 28)])
@pytest.mark.parametrize("capture_output", [False, True])
def test_hamming_conversion_in_subgraphs(source, target, capture_output):
    window = make_window_model(source, 0, TensorProto.FLOAT)
    inputs = [
        *window.graph.input,
        helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
    ]
    if capture_output:
        # A downstream If captures the output that the adapter replaces.
        branch = helper.make_graph(
            [helper.make_node("Identity", ["y"], ["branch_output"])],
            "capture",
            [],
            [helper.make_tensor_value_info("branch_output", TensorProto.FLOAT, [None])],
        )
        nodes = [*window.graph.node]
    else:
        # A HammingWindow inside If captures the outer graph's scalar size.
        branch = helper.make_graph(
            [*window.graph.node], "window", [], [*window.graph.output]
        )
        nodes = []
    nodes.append(
        helper.make_node(
            "If", ["condition"], ["result"], then_branch=branch, else_branch=branch
        )
    )
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "nested",
            inputs,
            [helper.make_tensor_value_info("result", TensorProto.FLOAT, [None])],
        ),
        opset_imports=[helper.make_opsetid("", source)],
    )
    converted = version_converter.convert_version(model, target)
    onnx.checker.check_model(converted, full_check=True)
    before = ReferenceEvaluator(model, new_ops=[HammingWindow])
    after = ReferenceEvaluator(converted)
    for condition in (False, True):
        feeds = {"size": np.array(7, dtype=np.int64), "condition": np.array(condition)}
        np.testing.assert_array_equal(
            after.run(None, feeds)[0], before.run(None, feeds)[0]
        )
