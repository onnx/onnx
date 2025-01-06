# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None  # type: ignore[assignment]

from onnx._custom_element_types import (
    bfloat16,
    float4e2m1,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
    int4,
    uint4,
)

_SUPPORTED_TYPES = (
    (bfloat16, "bfloat16", "bfloat16"),
    (float4e2m1, "float4e2m1", "float4_e2m1"),
    (float8e4m3fn, "e4m3fn", "float8_e4m3fn"),
    (float8e4m3fnuz, "e4m3fnuz", "float8_e4m3fnuz"),
    (float8e5m2, "e5m2", "float8_e5m2"),
    (float8e5m2fnuz, "e5m2fnuz", "float8_e5m2fnuz"),
    (int4, "int4", "int4"),
    (uint4, "uint4", "uint4"),
)

_ONNX_NAME_TO_ML_DTYPE_NAME = {elem[1]: elem[2] for elem in _SUPPORTED_TYPES}


def convert_from_ml_dtypes(array: np.ndarray) -> np.ndarray:
    """Detects the type and changes into one of the ONNX
    defined custom types when ``ml_dtypes`` is installed.

    Args:
        array: Numpy array with a dtype from ml_dtypes.

    Returns:
        Numpy array viewed with a dtype from ONNX custom types.
    """
    if ml_dtypes is None:
        return array
    for dtype, _, ml_name in _SUPPORTED_TYPES:
        ml_dtype = getattr(ml_dtypes, ml_name, None)
        if ml_dtype == array.dtype:
            return array.view(dtype=dtype)
    return array


def convert_to_ml_dtypes(array: np.ndarray) -> np.ndarray:
    """Detects the type and changes into one of the type
    defined in ``ml_dtypes`` if installed.

    Args:
        array: Numpy array with a native dtype or custom ONNX dtype.

    Returns:
        Numpy array with a dtype from ml_dtypes.
    """
    ml_dtype_name = _ONNX_NAME_TO_ML_DTYPE_NAME.get(array.dtype.descr[0][0], None)
    if ml_dtype_name is None:
        # The type is not a custom type
        return array
    if ml_dtypes is None:
        raise RuntimeError(
            f"ml_dtypes is not installed and the tensor cannot "
            f"be converted into ml_dtypes.{ml_dtype_name}"
        )
    return array.view(dtype=getattr(ml_dtypes, ml_dtype_name))
