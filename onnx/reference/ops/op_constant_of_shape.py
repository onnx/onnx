# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class ConstantOfShape(OpRun):
    def _run(self, data, value: np.array | None = None):
        if self.value is None:
            value = np.array(0, dtype=np.float32)
        else:
            if self.value.size != 1:
                raise ValueError(
                    f"Operator ConstantOfShape ({self.onnx_node.name!r}) expects a single element tensor as value, but the size of 'value' is {len(self.value)}."
                )
            value = self.value.flat[0]

        try:
            res = np.full(tuple(data), value)
        except TypeError as e:
            raise RuntimeError(
                f"Unable to create a constant of shape {data!r} with value {value!r} "
                f"(raw value={self.value!r})."
            ) from e
        return (res,)
