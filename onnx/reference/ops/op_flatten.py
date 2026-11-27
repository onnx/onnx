# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnary


class Flatten(OpRunUnary):
    def _run(self, x, axis=None):
        i = axis or self.axis
        shape = x.shape
        # Compute both dimensions explicitly: numpy cannot infer -1 when
        # the input has no elements (e.g. an empty batch).
        new_shape = (int(np.prod(shape[:i])), int(np.prod(shape[i:])))
        return (x.reshape(new_shape),)
