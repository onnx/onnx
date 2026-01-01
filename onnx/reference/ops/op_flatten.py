# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnary


class Flatten(OpRunUnary):
    def _run(self, x, axis=None):
        xp = self._get_array_api_namespace(x)
        i = axis or self.axis
        shape = x.shape
        # Use numpy for shape calculation
        new_shape = (1, -1) if i == 0 else (int(np.prod(shape[:i])), -1)
        return (xp.reshape(x, new_shape),)