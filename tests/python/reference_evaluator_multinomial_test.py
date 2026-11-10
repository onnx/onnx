# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator


def _evaluator(x, opset=22, **attributes):
    graph = helper.make_graph(
        [helper.make_node("Multinomial", ["X"], ["Y"], **attributes)],
        "multinomial",
        [
            helper.make_tensor_value_info(
                "X", helper.np_dtype_to_tensor_dtype(x.dtype), x.shape
            )
        ],
        [
            helper.make_tensor_value_info(
                "Y",
                attributes.get("dtype", TensorProto.INT32),
                [x.shape[0], attributes.get("sample_size", 1)],
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    checker.check_model(model, full_check=True)
    return ReferenceEvaluator(model)


@pytest.mark.parametrize("opset", [7, 22])
@pytest.mark.parametrize(
    "input_dtype", [np.float16, np.float32, np.float64, ml_dtypes.bfloat16]
)
@pytest.mark.parametrize("output_dtype", [TensorProto.INT32, TensorProto.INT64])
def test_multinomial_types_and_zero_probability(opset, input_dtype, output_dtype):
    if opset < 22 and input_dtype == ml_dtypes.bfloat16:
        pytest.skip("Bfloat16 was introduced in opset 22.")
    x = np.array([[-np.inf, 0, -np.inf], [0, -np.inf, -np.inf]], dtype=input_dtype)
    original = x.copy()
    evaluator = _evaluator(x, opset, dtype=output_dtype, sample_size=17, seed=42.0)
    (result,) = evaluator.run(None, {"X": x})
    assert result.dtype == helper.tensor_dtype_to_np_dtype(output_dtype)
    assert result.shape == (2, 17)
    assert_array_equal(result, [[1] * 17, [0] * 17])
    assert_array_equal(x, original)


@pytest.mark.parametrize("opset", [7, 22])
@pytest.mark.parametrize("seed", [0.0, 42.0, 42.5])
def test_multinomial_seeded_distributions(opset, seed):
    # This oracle checks this reference implementation's RNG convention, not
    # a portable sequence required of other ONNX implementations.
    probabilities = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    x = np.log(probabilities)
    original = x.copy()
    x.setflags(write=False)
    evaluator = _evaluator(x, opset, sample_size=1000, seed=seed)
    state = np.random.RandomState(int(seed))
    expected = np.array(
        [
            np.searchsorted(np.cumsum(row), state.uniform(size=1000), side="right")
            for row in probabilities
        ],
        dtype=np.int32,
    )
    (result,) = evaluator.run(None, {"X": x})
    assert_array_equal(result, expected)
    assert_array_equal(evaluator.run(None, {"X": x})[0], expected)
    assert_array_equal(x, original)


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, ml_dtypes.bfloat16]
)
def test_multinomial_large_values_and_shift_invariance(dtype):
    x = np.array([[0, -2, -4], [-4, 0, -2]], dtype=dtype)
    shifted = (x.astype(np.float64) + 128).astype(dtype)
    first = _evaluator(x, sample_size=100, seed=7.0).run(None, {"X": x})[0]
    second = _evaluator(shifted, sample_size=100, seed=7.0).run(None, {"X": shifted})[0]
    assert_array_equal(first, second)
    # Subtraction must not overflow in the input dtype. Equal large logits
    # must also be normalized without exponentiating the original values.
    limit = float(
        ml_dtypes.finfo(dtype).max
        if dtype == ml_dtypes.bfloat16
        else np.finfo(dtype).max
    )
    extreme = np.array([[limit, -limit], [limit, limit]], dtype=dtype)
    with np.errstate(all="raise"):
        result = _evaluator(extreme, sample_size=10, seed=7.0).run(
            None, {"X": extreme}
        )[0]
    assert_array_equal(result[0], np.zeros(10, dtype=np.int32))
    state = np.random.RandomState(7)
    state.uniform(size=10)
    assert_array_equal(result[1], (state.uniform(size=10) >= 0.5).astype(np.int32))


@pytest.mark.parametrize("seed", [None, float("nan"), 0.0, 42.5])
def test_multinomial_defaults(seed):
    x = np.zeros((3, 1), dtype=np.float32)
    attributes = {} if seed is None else {"seed": seed}
    result = _evaluator(x, **attributes).run(None, {"X": x})[0]
    assert result.dtype == np.int32
    assert_array_equal(result, np.zeros((3, 1), dtype=np.int32))


@pytest.mark.parametrize("shape,sample_size", [((0, 3), 4), ((2, 3), 0), ((0, 3), 0)])
def test_multinomial_empty_output(shape, sample_size):
    x = np.zeros(shape, dtype=np.float32)
    result = _evaluator(x, sample_size=sample_size).run(None, {"X": x})[0]
    assert result.shape == (shape[0], sample_size)
    assert result.dtype == np.int32


@pytest.mark.parametrize("values", [[[np.nan, 0]], [[np.inf, 0]], [[-np.inf, -np.inf]]])
def test_multinomial_non_normalizable_input(values):
    x = np.array(values, dtype=np.float32)
    # The spec does not define a distribution for these inputs. The reference
    # implementation reports an error rather than inventing probabilities.
    with pytest.raises(ValueError, match="normaliz"):
        _evaluator(x).run(None, {"X": x})
