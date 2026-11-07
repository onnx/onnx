# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator


@pytest.mark.parametrize("group,channels_per_group", [(1, 1), (2, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_conv_transpose_grouped_bias(group, channels_per_group, with_bias):
    channels = group * channels_per_group
    shape = [2, channels, 1, 1]
    inputs = [
        make_tensor_value_info("X", TensorProto.FLOAT, shape),
        make_tensor_value_info(
            "W", TensorProto.FLOAT, [channels, channels_per_group, 1, 1]
        ),
    ]
    feeds = {
        "X": np.arange(2 * channels, dtype=np.float32).reshape(shape),
        "W": np.ones((channels, channels_per_group, 1, 1), dtype=np.float32),
    }
    if with_bias:
        inputs.append(make_tensor_value_info("B", TensorProto.FLOAT, [channels]))
        feeds["B"] = np.arange(1, channels + 1, dtype=np.float32)
    node = make_node("ConvTranspose", list(feeds), ["Y"], group=group)
    graph = make_graph(
        [node], "g", inputs, [make_tensor_value_info("Y", TensorProto.FLOAT, shape)]
    )
    model = make_model(graph, opset_imports=[make_opsetid("", 11)])
    check_model(model)

    group_sums = feeds["X"].reshape(2, group, channels_per_group).sum(axis=2)
    expected = np.repeat(group_sums, channels_per_group, axis=1)
    if with_bias:
        expected += feeds["B"]
    got = ReferenceEvaluator(model).run(None, feeds)[0]
    assert_array_equal(got, expected.reshape(shape))
    assert got.dtype == np.float32
