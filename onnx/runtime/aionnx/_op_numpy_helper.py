# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore
from scipy.sparse import coo_matrix  # type: ignore


def numpy_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Implements a matmul product. See :func:`numpy.matmul`.
    Handles sparse matrices.
    """
    try:
        if isinstance(a, coo_matrix) or isinstance(b, coo_matrix):
            return np.dot(a, b)
        if len(a.shape) <= 2 and len(b.shape) <= 2:
            return np.dot(a, b)
        return np.matmul(a, b)
    except ValueError as e:
        raise ValueError(f"Unable to multiply shapes {a.shape!r}, {b.shape!r}.") from e
