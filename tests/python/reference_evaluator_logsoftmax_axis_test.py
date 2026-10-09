# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator

# Literal log probabilities from independent high-precision mathematical values.
# Arrays describe the feature groups explicitly, without deriving an oracle
# from the reference operator or a NumPy reduction.

_INPUT_LITERAL = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]

_FLAT4 = [
    [
        [-3.4401896985611953, -2.4401896985611953],
        [-1.4401896985611953, -0.44018969856119533],
    ],
    [
        [-3.4401896985611953, -2.4401896985611953],
        [-1.4401896985611953, -0.44018969856119533],
    ],
]

_FLAT8 = [
    [
        [-7.458339626479005, -6.458339626479005],
        [-5.458339626479005, -4.458339626479005],
    ],
    [
        [-3.458339626479005, -2.458339626479005],
        [-1.4583396264790052, -0.45833962647900506],
    ],
]

_PAIR1 = [
    [
        [-1.3132616875182228, -0.3132616875182228],
        [-1.3132616875182228, -0.3132616875182228],
    ],
    [
        [-1.3132616875182228, -0.3132616875182228],
        [-1.3132616875182228, -0.3132616875182228],
    ],
]

_PAIR2 = [
    [
        [-2.1269280110429727, -2.1269280110429727],
        [-0.1269280110429725, -0.1269280110429725],
    ],
    [
        [-2.1269280110429727, -2.1269280110429727],
        [-0.1269280110429725, -0.1269280110429725],
    ],
]

_PAIR4 = [
    [
        [-4.0181499279178094, -4.0181499279178094],
        [-4.0181499279178094, -4.0181499279178094],
    ],
    [
        [-0.01814992791780974, -0.01814992791780974],
        [-0.01814992791780974, -0.01814992791780974],
    ],
]


_CASES = [
    pytest.param("opset11_axis1_flatten4", 11, 1, _FLAT4, id="opset11_axis1_flatten4"),
    pytest.param(
        "opset11_negative2_flatten4", 11, -2, _FLAT4, id="opset11_negative2_flatten4"
    ),
    pytest.param("opset11_axis0_flatten8", 11, 0, _FLAT8, id="opset11_axis0_flatten8"),
    pytest.param(
        "opset11_negative3_flatten8", 11, -3, _FLAT8, id="opset11_negative3_flatten8"
    ),
    pytest.param("opset11_default_axis1", 11, None, _FLAT4, id="opset11_default_axis1"),
    pytest.param("opset11_axis2_control", 11, 2, _PAIR1, id="opset11_axis2_control"),
    pytest.param(
        "opset11_negative1_control", 11, -1, _PAIR1, id="opset11_negative1_control"
    ),
    pytest.param("opset13_axis1_control", 13, 1, _PAIR2, id="opset13_axis1_control"),
    pytest.param("opset13_axis0_control", 13, 0, _PAIR4, id="opset13_axis0_control"),
    pytest.param(
        "opset13_default_negative1_control",
        13,
        None,
        _PAIR1,
        id="opset13_default_negative1_control",
    ),
]


@pytest.mark.parametrize("name,opset,axis,expected", _CASES)
def test_logsoftmax_opset_axis(
    name: str, opset: int, axis: int | None, expected: list[list[list[float]]]
) -> None:
    x = np.array(_INPUT_LITERAL, dtype=np.float32)
    attributes = {} if axis is None else {"axis": axis}
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("LogSoftmax", ["X"], ["Y"], **attributes)],
            name,
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2, 2])],
        ),
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=10,
    )
    checker.check_model(model, full_check=True)
    actual = ReferenceEvaluator(model).run(None, {"X": x})[0]
    np.testing.assert_allclose(
        actual, np.array(expected, dtype=np.float32), rtol=1e-6, atol=2e-7
    )
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert np.isfinite(actual).all()
