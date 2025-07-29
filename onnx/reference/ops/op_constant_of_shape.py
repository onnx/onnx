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
            value = self.value.item()

        try:
            res = np.full(tuple(data), value)
        except TypeError as e:
            raise RuntimeError(
                f"Unable to create a constant of shape {data!r} with value {self.cst!r} "
                f"(raw value={value!r})."
            ) from e
        return (res,)
