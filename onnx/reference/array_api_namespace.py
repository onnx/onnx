# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
"""Array API namespace wrapper for the reference runtime.

This module provides a unified interface for array operations using the Array API standard.
It supports numpy, cupy, jax, pytorch (torch), and mlx backends through array-api-compat.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import array_api_compat
    ARRAY_API_COMPAT_AVAILABLE = True
except ImportError:
    ARRAY_API_COMPAT_AVAILABLE = False


def get_array_api_namespace(array: Any) -> Any:
    """Get the array API namespace for the given array.
    
    Args:
        array: An array object (numpy, cupy, jax, torch, mlx, etc.)
        
    Returns:
        The array API namespace for operations on this array type.
        Falls back to numpy for non-array types or if array-api-compat is not available.
    """
    if not ARRAY_API_COMPAT_AVAILABLE:
        return np
    
    # Use array-api-compat to get the appropriate namespace
    try:
        return array_api_compat.array_namespace(array)
    except (TypeError, ValueError):
        # Fallback to numpy for unsupported types
        return np


def is_array_api_obj(obj: Any) -> bool:
    """Check if an object is an array API compatible object.
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object is an array, False otherwise
    """
    if isinstance(obj, np.ndarray):
        return True
    
    if not ARRAY_API_COMPAT_AVAILABLE:
        return False
    
    try:
        array_api_compat.array_namespace(obj)
        return True
    except (TypeError, ValueError):
        return False


def asarray(obj: Any, dtype: Any = None, device: Any = None, xp: Any = None) -> Any:
    """Convert object to array using the appropriate namespace.
    
    Args:
        obj: Object to convert
        dtype: Target dtype
        device: Target device (if supported)
        xp: Array namespace to use (if None, inferred from obj)
        
    Returns:
        Array object
    """
    if xp is None:
        if is_array_api_obj(obj):
            xp = get_array_api_namespace(obj)
        else:
            xp = np
    
    # Use asarray if available, otherwise use array
    if hasattr(xp, 'asarray'):
        if dtype is not None:
            return xp.asarray(obj, dtype=dtype)
        return xp.asarray(obj)
    else:
        if dtype is not None:
            return xp.array(obj, dtype=dtype)
        return xp.array(obj)


def convert_to_numpy(array: Any) -> np.ndarray:
    """Convert an array API compatible array to numpy.
    
    Args:
        array: Array to convert
        
    Returns:
        Numpy array
    """
    if isinstance(array, np.ndarray):
        return array
    
    # Try to use device() and to_device() for arrays with device support
    xp = get_array_api_namespace(array)
    
    # For PyTorch
    if hasattr(array, 'cpu') and hasattr(array, 'numpy'):
        return array.cpu().numpy()
    
    # For JAX
    if hasattr(array, '__array__'):
        return np.asarray(array)
    
    # For CuPy
    if hasattr(array, 'get'):
        return array.get()
    
    # For MLX
    if hasattr(xp, '__name__') and 'mlx' in xp.__name__.lower():
        # MLX arrays can be converted to numpy via __array__
        if hasattr(array, '__array__'):
            return np.asarray(array)
    
    # Generic fallback
    return np.asarray(array)
