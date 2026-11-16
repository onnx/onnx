# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import TensorProto, checker, helper, numpy_helper, version_converter
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun


class Upsample(OpRun):
    """Independent integer-nearest oracle for the attribute-based operator."""

    op_domain = ""

    def _run(self, x, mode=None, scales=None):
        assert mode == "nearest"
        result = x
        for axis, scale in enumerate(scales):
            assert int(scale) == scale
            result = np.repeat(result, int(scale), axis=axis)
        return (result,)


def make_model(provider, sharing):
    scales = numpy_helper.from_array(
        np.array([1, 1, 2, 2], dtype=np.float32), name="scales"
    )
    nodes = []
    initializers = []
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 2, 2])]
    if provider == "constant":
        nodes.append(helper.make_node("Constant", [], ["scales"], value=scales))
    elif provider != "dynamic":
        initializers.append(scales)
    if provider in ("overridable", "dynamic"):
        inputs.append(helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4]))
    nodes.append(helper.make_node("Upsample", ["X", "scales"], ["Y"], mode="nearest"))
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 4])]
    if sharing == "node":
        nodes.append(helper.make_node("Identity", ["scales"], ["copy"]))
        outputs.append(helper.make_tensor_value_info("copy", TensorProto.FLOAT, [4]))
    elif sharing == "output":
        outputs.append(helper.make_tensor_value_info("scales", TensorProto.FLOAT, [4]))
    elif sharing == "upsample":
        nodes.append(
            helper.make_node("Upsample", ["X", "scales"], ["Z"], mode="nearest")
        )
        outputs.append(
            helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 1, 4, 4])
        )
    result = helper.make_model(
        helper.make_graph(nodes, "shared-scales", inputs, outputs, initializers),
        opset_imports=[helper.make_opsetid("", 9)],
        ir_version=8,
    )
    checker.check_model(result, full_check=True)
    return result


@pytest.mark.parametrize("provider", ["constant", "initializer"])
@pytest.mark.parametrize("sharing", ["none", "node", "output", "upsample"])
def test_preserve_shared_scales(provider, sharing):
    original = make_model(provider, sharing)
    before = original.SerializeToString()
    converted = version_converter.convert_version(original, 8)
    assert original.SerializeToString() == before
    checker.check_model(converted, full_check=True)
    data = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    expected = ReferenceEvaluator(original).run(None, {"X": data})
    actual = ReferenceEvaluator(converted, new_ops=[Upsample]).run(None, {"X": data})
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    if sharing in ("none", "upsample"):
        assert not converted.graph.initializer
        assert all(node.op_type != "Constant" for node in converted.graph.node)


@pytest.mark.parametrize("provider", ["overridable", "dynamic"])
def test_reject_nonconstant_scales(provider):
    original = make_model(provider, "none")
    before = original.SerializeToString()
    with pytest.raises(RuntimeError, match=r"overridable|unavailable input"):
        version_converter.convert_version(original, 8)
    assert original.SerializeToString() == before
