# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class BitCast(OpRun):
    def _run(self, x, to=None):  # type: ignore
        if to is None:
            raise ValueError("The 'to' attribute must be specified for BitCast.")
        
        # Map ONNX data types to numpy dtypes
        onnx_to_numpy_dtype = {
            1: np.float32,      # FLOAT
            2: np.uint8,        # UINT8
            3: np.int8,         # INT8
            5: np.int16,        # INT16
            6: np.int32,        # INT32
            7: np.int64,        # INT64
            9: np.bool_,        # BOOL
            10: np.float16,     # FLOAT16
            11: np.float64,     # DOUBLE
            12: np.uint32,      # UINT32
            13: np.uint64,      # UINT64
            16: np.dtype('float16'),  # BFLOAT16 (approximated as float16)
            # Note: numpy doesn't have native bfloat16, using float16 as approximation
            # 4-bit types and float8 types would require special handling
        }
        
        target_dtype = onnx_to_numpy_dtype.get(to)
        if target_dtype is None:
            raise ValueError(f"Unsupported target type: {to}")
        
        # Get element sizes in bytes
        input_itemsize = x.dtype.itemsize
        output_itemsize = np.dtype(target_dtype).itemsize
        
        if input_itemsize == output_itemsize:
            # Same size: simple view
            result = x.view(target_dtype)
        elif input_itemsize < output_itemsize:
            # Input smaller than output: need to combine elements
            # The last dimension must be divisible by the size ratio
            size_ratio = output_itemsize // input_itemsize
            
            if x.shape[-1] % size_ratio != 0:
                raise ValueError(
                    f"The last dimension ({x.shape[-1]}) must be divisible by "
                    f"the size ratio ({size_ratio}) when bitcasting from "
                    f"{x.dtype} to {target_dtype}"
                )
            
            # Flatten to 1D, view as target type, then reshape
            new_shape = list(x.shape[:-1]) + [x.shape[-1] // size_ratio]
            result = x.reshape(-1).view(target_dtype).reshape(new_shape)
        else:
            # Input larger than output: need to split elements  
            size_ratio = input_itemsize // output_itemsize
            
            # Flatten to 1D, view as target type, then reshape
            new_shape = list(x.shape[:-1]) + [x.shape[-1] * size_ratio] if len(x.shape) > 0 else [size_ratio]
            if len(x.shape) == 0:
                # Scalar input
                result = x.reshape(1).view(target_dtype).reshape(size_ratio)
            else:
                result = x.reshape(-1).view(target_dtype).reshape(new_shape)
        
        return (result,)
