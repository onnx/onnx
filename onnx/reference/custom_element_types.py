# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None  # type: ignore[assignment]

bfloat16 = np.dtype((np.uint16, {"bfloat16": (np.uint16, 0)}))
float8e4m3fn = np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)}))
float8e4m3fnuz = np.dtype((np.uint8, {"e4m3fnuz": (np.uint8, 0)}))
float8e5m2 = np.dtype((np.uint8, {"e5m2": (np.uint8, 0)}))
float8e5m2fnuz = np.dtype((np.uint8, {"e5m2fnuz": (np.uint8, 0)}))
uint4 = np.dtype((np.uint8, {"uint4": (np.uint8, 0)}))
int4 = np.dtype((np.int8, {"int4": (np.int8, 0)}))


def convert_from_ml_dtypes(tensor: np.ndarray) -> np.ndarray:
    """If package *ml_dtypes* is installed,
    this function detects the type and changes into one of the custom type
    defined in this module.
    The function returns tensor if numpy supports the type,
    raises an exception if ml_dtypes is not installed.

    Args:
        tensor: numpy array with a dtype from ml_dtypes

    Returns:
        numpy array
    """
    if not ml_dtypes:
        return tensor
    dt = tensor.dtype
    new_dt = None
    if dt == ml_dtypes.bfloat16:
        new_dt = bfloat16

    if new_dt:
        b = tensor.tobytes()
        return np.frombuffer(b, dtype=new_dt).reshape(tensor.shape)

    return tensor


def convert_to_ml_dtypes(tensor: np.ndarray) -> np.ndarray:
    """If package *ml_dtypes* is installed,
    this fonction changes dtype into the corresponding
    dtype so that numpy can do simple operations.
    The function returns tensor if numpy supports the type,
    raises an exception if ml_dtypes is not installed.

    Args:
        tensor: numpy array

    Returns:
        numpy array with a dtype from ml_dtypes
    """
    dt = tensor.dtype
    new_dt = None
    if dt == bfloat16 and tensor.dtype.descr[0][0] == "bfloat16":
        assert ml_dtypes, (
            f"ml_dtypes is not installed and the tensor cannot "
            f"be converted into ml_dtypes.{tensor.dtype.descr[0][0]}"
        )

        new_dt = ml_dtypes.bfloat16

    if new_dt:
        b = tensor.tobytes()
        return np.frombuffer(b, dtype=new_dt).reshape(tensor.shape)

    return tensor
