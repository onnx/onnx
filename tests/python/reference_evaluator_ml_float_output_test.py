# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
    np_dtype_to_tensor_dtype,
)
from onnx.reference import ReferenceEvaluator


def _run(node_inputs, x_list, **attrs):
    op_type = attrs.pop("op_type")
    node = make_node(op_type, node_inputs, ["Y"], domain="ai.onnx.ml", **attrs)
    graph = make_graph(
        [node],
        "g",
        [
            make_tensor_value_info(name, np_dtype_to_tensor_dtype(x.dtype), None)
            for name, x in zip(node_inputs, x_list, strict=True)
        ],
        [make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = make_model(
        graph, opset_imports=[make_opsetid("", 18), make_opsetid("ai.onnx.ml", 3)]
    )
    return ReferenceEvaluator(model).run(
        None, dict(zip(node_inputs, x_list, strict=True))
    )[0]


# The ai.onnx.ml operators below declare their output as tensor(float).
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
class TestReferenceEvaluatorMLFloatOutput:
    def test_scaler(self, dtype) -> None:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        y = _run(["X"], [x], op_type="Scaler", offset=[0.5], scale=[1.5])
        assert y.dtype == np.float32
        assert_allclose(y, (x.astype(np.float32) - 0.5) * 1.5)

    def test_linear_regressor(self, dtype) -> None:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        y = _run(
            ["X"],
            [x],
            op_type="LinearRegressor",
            coefficients=[0.5, 1.5, 2.5],
            intercepts=[0.25],
            targets=1,
        )
        assert y.dtype == np.float32
        assert_allclose(y, x.astype(np.float32) @ [[0.5], [1.5], [2.5]] + 0.25)

    def test_feature_vectorizer(self, dtype) -> None:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        y = _run(["X"], [x], op_type="FeatureVectorizer", inputdimensions=[3])
        assert y.dtype == np.float32
        assert_allclose(y, x)
