# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunBinaryNum


def numpy_matmul(a, b):  # type: ignore
    """
    Implements a matmul product. See :func:`numpy.matmul`.
    Handles sparse matrices.
    """
    try:
        if len(a.shape) <= 2 and len(b.shape) <= 2:
            return np.dot(a, b)
        return np.matmul(a, b)
    except ValueError as e:
        raise ValueError(f"Unable to multiply shapes {a.shape!r}, {b.shape!r}.") from e


class MatMul(OpRunBinaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNum.__init__(self, onnx_node, run_params)

    def _run(self, a, b):  # type: ignore
        return (numpy_matmul(a, b),)
