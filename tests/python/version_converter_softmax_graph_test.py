# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import TensorProto, checker, helper, numpy_helper, version_converter
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun


def legacy_normalize(x, axis, logarithmic):
    """Opset <= 12 normalizes over dimensions from axis to the last one."""
    axis %= x.ndim
    flat = x.reshape(int(np.prod(x.shape[:axis])), -1)
    shifted = flat - flat.max(axis=1, keepdims=True)
    result = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    if not logarithmic:
        result = np.exp(result)
    return (result.reshape(x.shape),)


class Softmax(OpRun):
    op_domain = ""

    def _run(self, x, axis=1):
        return legacy_normalize(x, axis, False)


class LogSoftmax(OpRun):
    op_domain = ""

    def _run(self, x, axis=1):
        return legacy_normalize(x, axis, True)


def value(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("layout", ["flatten", "reshape", "both"])
@pytest.mark.parametrize("shared", [False, True])
def test_preserve_neighboring_shape_operators(op, layout, shared):
    before = layout in ("flatten", "both")
    after = layout in ("reshape", "both")
    shape = [2, 3, 4] if before else [2, 12]
    nodes = []
    initializers = []
    outputs = []
    if before:
        nodes.append(helper.make_node("Flatten", ["X"], ["F"], axis=1))
        if shared:
            outputs.append(value("F", [2, 12]))
    nodes.append(helper.make_node(op, ["F" if before else "X"], ["S"], axis=-1))
    if after:
        initializers.append(
            numpy_helper.from_array(np.array([4, 6], dtype=np.int64), name="shape")
        )
        nodes.append(helper.make_node("Reshape", ["S", "shape"], ["Y"]))
        outputs.append(value("Y", [4, 6]))
        if shared:
            outputs.append(value("S", [2, 12]))
    else:
        outputs.append(value("S", [2, 12]))
    original = helper.make_model(
        helper.make_graph(
            nodes, "neighbors", [value("X", shape)], outputs, initializers
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    checker.check_model(original, full_check=True)
    serialized = original.SerializeToString()
    converted = version_converter.convert_version(original, 12)
    assert original.SerializeToString() == serialized
    checker.check_model(converted, full_check=True)
    assert [n.op_type for n in converted.graph.node] == [n.op_type for n in nodes]
    assert [o.name for o in converted.graph.output] == [o.name for o in outputs]
    data = np.arange(24, dtype=np.float32).reshape(shape) / 10
    expected = ReferenceEvaluator(original).run(None, {"X": data})
    actual = ReferenceEvaluator(converted, new_ops=[Softmax, LogSoftmax]).run(
        None, {"X": data}
    )
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("op", ["Softmax", "LogSoftmax"])
@pytest.mark.parametrize("axis", [0, 1, -2])
def test_softmax_roundtrip(op, axis):
    original = helper.make_model(
        helper.make_graph(
            [helper.make_node(op, ["X"], ["Y"], axis=axis)],
            "roundtrip",
            [value("X", [2, 3, 4])],
            [value("Y", [2, 3, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 12)],
    )
    checker.check_model(original, full_check=True)
    upgraded = version_converter.convert_version(original, 13)
    checker.check_model(upgraded, full_check=True)
    converted = version_converter.convert_version(upgraded, 12)
    checker.check_model(converted, full_check=True)
    assert [n.op_type for n in converted.graph.node] == [
        "Shape",
        "Flatten",
        op,
        "Reshape",
    ]
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
    expected = ReferenceEvaluator(original, new_ops=[Softmax, LogSoftmax]).run(
        None, {"X": data}
    )[0]
    actual = ReferenceEvaluator(converted, new_ops=[Softmax, LogSoftmax]).run(
        None, {"X": data}
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("op", ["Shape", "Flatten", "Reshape"])
def test_shape_operator_downgrade_rejects_bfloat16(op):
    inputs = ["X"]
    initializers = []
    if op == "Reshape":
        inputs.append("shape")
        initializers.append(
            numpy_helper.from_array(np.array([2, 2], dtype=np.int64), name="shape")
        )
    output = (
        value("Y", [2], TensorProto.INT64)
        if op == "Shape"
        else value("Y", [2, 2], TensorProto.BFLOAT16)
    )
    original = helper.make_model(
        helper.make_graph(
            [helper.make_node(op, inputs, ["Y"])],
            "bfloat16",
            [value("X", [2, 2], TensorProto.BFLOAT16)],
            [output],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    checker.check_model(original, full_check=True)
    with pytest.raises(RuntimeError, match="unallowed"):
        version_converter.convert_version(original, 12)
