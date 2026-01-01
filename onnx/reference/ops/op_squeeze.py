# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any


from onnx.reference.array_api_namespace import convert_to_numpy, asarray
from onnx.reference.op_run import OpRun


class Squeeze_1(OpRun):
    def _run(self, data, axes=None):
        xp = self._get_array_api_namespace(data)
        if isinstance(axes, Any):
            axes = tuple(axes)
        elif axes in [[], ()]:
            axes = None
        elif isinstance(axes, list):
            axes = tuple(axes)
        
        # Convert to numpy for squeeze operation (not in array API standard yet)
        data_np = convert_to_numpy(data)
        
        if isinstance(axes, (tuple, list)):
            sq = data_np
            for a in reversed(axes):
                sq = np.squeeze(sq, axis=a)
        else:
            sq = np.squeeze(data_np, axis=axes)
        
        # Convert back to original array type
        sq = asarray(sq, xp=xp)
        return (sq,)


class Squeeze_11(Squeeze_1):
    pass


class Squeeze_13(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        self.axes = None

    def _run(self, data, axes=None):
        xp = self._get_array_api_namespace(data)
        
        # Convert to numpy for squeeze operation (not in array API standard yet)
        data_np = convert_to_numpy(data)
        
        if axes is not None:
            if hasattr(axes, "__iter__"):
                sq = np.squeeze(data_np, axis=tuple(axes))
            else:
                sq = np.squeeze(data_np, axis=axes)
        else:
            sq = np.squeeze(data_np)
        
        # Convert back to original array type
        sq = asarray(sq, xp=xp)
        return (sq,)
