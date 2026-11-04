# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import TensorProto, checker, helper, numpy_helper, version_converter
from onnx.reference import ReferenceEvaluator

CASES = [
    ("ReduceSum", 13, 12, [2, 2], [0], [[1, 2]]),
    ("ReduceMean", 18, 17, [2, 2], [0], [[1, 2]]),
    ("Squeeze", 13, 12, [1, 1], [0], [[1]]),
    ("Unsqueeze", 13, 12, [2], [0], [[1, 2]]),
    ("Split", 13, 12, [4], [2, 2], [[2], [2]]),
]


def make_model(case, provider, shared=False, omitted=False):
    op, source, target, shape, parameter, output_shapes = case
    del target
    nodes = []
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)]
    initializers = []
    parameter_tensor = numpy_helper.from_array(
        np.asarray(parameter, dtype=np.int64), name="parameter"
    )
    if not omitted:
        if provider == "constant":
            nodes.append(
                helper.make_node("Constant", [], ["parameter"], value=parameter_tensor)
            )
        elif provider != "dynamic":
            initializers.append(parameter_tensor)
        if provider in ("overridable", "dynamic"):
            inputs.append(
                helper.make_tensor_value_info(
                    "parameter", TensorProto.INT64, [len(parameter)]
                )
            )
    names = [f"Y{i}" for i in range(len(output_shapes))]
    args = ["X", "" if omitted else "parameter"]
    nodes.append(helper.make_node(op, args, names))
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, output_shape)
        for name, output_shape in zip(names, output_shapes, strict=True)
    ]
    if shared:
        outputs.append(
            helper.make_tensor_value_info(
                "parameter", TensorProto.INT64, [len(parameter)]
            )
        )
    result = helper.make_model(
        helper.make_graph(nodes, "input-safety", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", source)],
        ir_version=8,
    )
    checker.check_model(result, full_check=True)
    return result


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("provider", ["overridable", "dynamic"])
def test_reject_nonconstant_input(case, provider):
    original = make_model(case, provider)
    before = original.SerializeToString()
    with pytest.raises(
        RuntimeError, match="dynamic inputs and overridable initializers"
    ):
        version_converter.convert_version(original, case[2])
    assert original.SerializeToString() == before


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("provider", ["constant", "initializer"])
@pytest.mark.parametrize("shared", [False, True])
def test_constant_input_preserves_values_and_other_uses(case, provider, shared):
    original = make_model(case, provider, shared)
    before = original.SerializeToString()
    converted = version_converter.convert_version(original, case[2])
    assert original.SerializeToString() == before
    checker.check_model(converted, full_check=True)
    data = np.arange(np.prod(case[3]), dtype=np.float32).reshape(case[3])
    expected = ReferenceEvaluator(original).run(None, {"X": data})
    actual = ReferenceEvaluator(converted).run(None, {"X": data})
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference)
    if shared:
        assert converted.graph.output[-1].name == "parameter"
    else:
        assert not converted.graph.initializer
        assert all(node.op_type != "Constant" for node in converted.graph.node)


@pytest.mark.parametrize(
    "case",
    [
        ("ReduceSum", 13, 12, [2, 2], [], [[1, 1]]),
        ("ReduceMean", 18, 17, [2, 2], [], [[1, 1]]),
        ("Squeeze", 13, 12, [1, 2, 1], [], [[2]]),
        ("Split", 13, 12, [4], [], [[2], [2]]),
    ],
)
def test_explicitly_omitted_optional_input(case):
    original = make_model(case, "dynamic", omitted=True)
    converted = version_converter.convert_version(original, case[2])
    checker.check_model(converted, full_check=True)
    assert list(converted.graph.node[0].input) == ["X"]
    data = np.arange(np.prod(case[3]), dtype=np.float32).reshape(case[3])
    expected = ReferenceEvaluator(original).run(None, {"X": data})
    actual = ReferenceEvaluator(converted).run(None, {"X": data})
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference)
