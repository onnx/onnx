# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from onnx.helper import make_node
from onnx.reference import ReferenceEvaluator


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize(
    ("image_shape", "block_shape", "input_shape", "output_shape"),
    [
        ([3, 3], [2, 2], (0, 8, 4), (0, 2, 3, 3)),
        ([2, 3, 4], [1, 2, 2], (0, 4, 12), (0, 1, 2, 3, 4)),
    ],
)
def test_col2im_empty_batch(dtype, image_shape, block_shape, input_shape, output_shape):
    node = make_node("Col2Im", ["X", "image_shape", "block_shape"], ["Y"])
    result = ReferenceEvaluator(node).run(
        None,
        {
            "X": np.empty(input_shape, dtype=dtype),
            "image_shape": np.array(image_shape, dtype=np.int64),
            "block_shape": np.array(block_shape, dtype=np.int64),
        },
    )[0]
    assert isinstance(result, np.ndarray)
    assert result.shape == output_shape
    assert result.dtype == dtype


def test_col2im_nonempty_batch_control():
    node = make_node("Col2Im", ["X", "image_shape", "block_shape"], ["Y"])
    data = np.arange(4, dtype=np.float32).reshape(1, 1, 4)
    result = ReferenceEvaluator(node).run(
        None,
        {
            "X": data,
            "image_shape": np.array([2, 2], dtype=np.int64),
            "block_shape": np.array([1, 1], dtype=np.int64),
        },
    )[0]
    np.testing.assert_array_equal(result, data.reshape(1, 1, 2, 2))
