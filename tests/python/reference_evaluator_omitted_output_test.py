# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorOmittedOutput:
    def test_omitted_output_does_not_fill_omitted_input(self) -> None:
        # Dropout's second output (the mask) is omitted with an empty name.
        # Clip's first optional input ('min') is omitted with the same empty name,
        # so it must stay unset instead of receiving the mask.
        graph = make_graph(
            [
                make_node("Dropout", ["X"], ["D", ""]),
                make_node("Clip", ["D", "", "M"], ["Y"]),
            ],
            "g",
            [
                make_tensor_value_info("X", TensorProto.FLOAT, [3]),
                make_tensor_value_info("M", TensorProto.FLOAT, []),
            ],
            [make_tensor_value_info("Y", TensorProto.FLOAT, [3])],
        )
        model = make_model(graph, opset_imports=[make_opsetid("", 18)])
        x = np.array([-5, 1, 9], dtype=np.float32)
        (y,) = ReferenceEvaluator(model).run(
            None, {"X": x, "M": np.array(4, dtype=np.float32)}
        )
        assert_allclose(y, np.minimum(x, 4))
