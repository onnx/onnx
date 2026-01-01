# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


def reshape_reference_implementation(data, shape, allowzero: int, xp):
    """Reference implementation of reshape using array API.

    Args:
        data: Input array
        shape: Target shape
        allowzero: If 0, zeros in shape are replaced with corresponding dim size
        xp: Array API namespace
    """
    # Convert shape to a list/tuple we can work with
    if hasattr(shape, "tolist"):
        new_shape = list(shape.tolist())
    else:
        new_shape = list(shape)

    # Replace zeros with corresponding dim size if allowzero is 0
    if allowzero == 0:
        for i in range(len(new_shape)):
            if new_shape[i] == 0:
                new_shape[i] = data.shape[i]

    return xp.reshape(data, tuple(new_shape))


class CommonReshape(OpRun):
    def _run(self, data, shape):
        xp = self._get_array_api_namespace(data)
        return (reshape_reference_implementation(data, shape, 0, xp),)


class Reshape_5(CommonReshape):
    pass


class Reshape_14(CommonReshape):
    def _run(self, data, shape, allowzero=None):
        xp = self._get_array_api_namespace(data)
        if allowzero is None:
            allowzero = getattr(self, "allowzero", 0) == 1
        else:
            allowzero = allowzero == 1
        return (reshape_reference_implementation(data, shape, allowzero, xp),)
