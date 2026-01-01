# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.array_api_namespace import convert_to_numpy, asarray
from onnx.reference.op_run import OpRun


class Unsqueeze_1(OpRun):
    def _run(self, data, axes=None):
        xp = self._get_array_api_namespace(data)
        
        # Convert to numpy for unsqueeze (expand_dims not in array API standard yet)
        data_np = convert_to_numpy(data)
        
        if isinstance(axes, np.ndarray):
            axes = tuple(axes)
        elif axes in ([], ()):
            axes = None
        elif isinstance(axes, list):
            axes = tuple(axes)
        
        if isinstance(axes, (tuple, list)):
            sq = data_np
            for a in axes:
                sq = np.expand_dims(sq, axis=a)
        else:
            raise TypeError("axes cannot be None for operator Unsqueeze (Unsqueeze_1).")
        
        # Convert back to original array type
        return (asarray(sq, xp=xp),)


class Unsqueeze_11(Unsqueeze_1):
    pass


class Unsqueeze_13(OpRun):
    def _run(self, data, axes=None):
        xp = self._get_array_api_namespace(data)
        
        # Convert to numpy for unsqueeze (expand_dims not in array API standard yet)
        data_np = convert_to_numpy(data)
        
        if axes is not None:
            if hasattr(axes, "__iter__") and len(axes.shape) > 0:
                try:
                    sq = np.expand_dims(data_np, axis=tuple(axes))
                except TypeError:
                    # numpy 1.18 supports axes as a tuple
                    if len(axes) == 1:
                        sq = np.expand_dims(data_np, axis=tuple(axes)[0])
                    else:
                        sq = data_np
                        for a in reversed(axes):
                            sq = np.expand_dims(sq, axis=a)
            else:
                sq = np.expand_dims(data_np, axis=axes)
        else:
            raise RuntimeError(
                "axes cannot be None for operator Unsqueeze (Unsqueeze_13)."
            )
        
        # Convert back to original array type
        return (asarray(sq, xp=xp),)