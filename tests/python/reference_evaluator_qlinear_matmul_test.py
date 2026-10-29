# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import checker
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
    np_dtype_to_tensor_dtype,
)
from onnx.reference import ReferenceEvaluator


class TestReferenceEvaluatorQLinearMatMul:
    @pytest.mark.parametrize(
        "opset,scale_dtype", [(10, np.float32), (21, np.float32), (21, np.float16)]
    )
    @pytest.mark.parametrize(
        "dtype,zero_point",
        [
            (np.int8, -1),
            (np.int8, 0),
            (np.int8, 1),
            (np.int8, 2),
            (np.uint8, 0),
            (np.uint8, 1),
            (np.uint8, 2),
        ],
    )
    @pytest.mark.parametrize("parameter_shape", [(), (1,)])
    def test_round_before_adding_zero_point(
        self, opset, scale_dtype, dtype, zero_point, parameter_shape
    ):
        values = [1, 3, 5, -1, -3, -5, 2, -2, 0]
        feeds = {
            "a": np.array([[v + 8 for v in values]], dtype=dtype),
            "a_scale": np.full(parameter_shape, 0.5, dtype=scale_dtype),
            "a_zero_point": np.full(parameter_shape, 8, dtype=dtype),
            "b": np.eye(len(values), dtype=dtype),
            "b_scale": np.full(parameter_shape, 1, dtype=scale_dtype),
            "b_zero_point": np.full(parameter_shape, 0, dtype=dtype),
            "y_scale": np.full(parameter_shape, 1, dtype=scale_dtype),
            "y_zero_point": np.full(parameter_shape, zero_point, dtype=dtype),
        }
        qmin, qmax = np.iinfo(dtype).min, np.iinfo(dtype).max
        expected = np.array(
            [[min(qmax, max(qmin, round(v / 2) + zero_point)) for v in values]],
            dtype=dtype,
        )
        model = make_model(
            make_graph(
                [make_node("QLinearMatMul", list(feeds), ["Y"])],
                "qlinear_matmul",
                [
                    make_tensor_value_info(
                        name, np_dtype_to_tensor_dtype(value.dtype), value.shape
                    )
                    for name, value in feeds.items()
                ],
                [
                    make_tensor_value_info(
                        "Y", np_dtype_to_tensor_dtype(np.dtype(dtype)), expected.shape
                    )
                ],
            ),
            opset_imports=[make_opsetid("", opset)],
        )
        checker.check_model(model, full_check=True)
        (got,) = ReferenceEvaluator(model).run(None, feeds)

        assert got.dtype == expected.dtype
        assert_array_equal(got, expected)

    @pytest.mark.parametrize("dtype", [np.int8, np.uint8])
    def test_saturation_after_rounding(self, dtype):
        feeds = {
            "a": np.array([[0, 127]], dtype=dtype),
            "a_scale": np.array(1, dtype=np.float32),
            "a_zero_point": np.array(64, dtype=dtype),
            "b": np.array([[127, 0], [0, 127]], dtype=dtype),
            "b_scale": np.array(1, dtype=np.float32),
            "b_zero_point": np.array(0, dtype=dtype),
            "y_scale": np.array(1, dtype=np.float32),
            "y_zero_point": np.array(1, dtype=dtype),
        }
        node = make_node("QLinearMatMul", list(feeds), ["Y"])
        expected = np.array([[np.iinfo(dtype).min, np.iinfo(dtype).max]], dtype=dtype)

        (got,) = ReferenceEvaluator(node).run(None, feeds)

        assert got.dtype == expected.dtype
        assert_array_equal(got, expected)
