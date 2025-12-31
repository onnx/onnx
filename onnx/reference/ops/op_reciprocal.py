# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Reciprocal(OpRunUnaryNum):
    def _run(self, x):
        xp = self._get_array_api_namespace(x)
        with np.errstate(divide="ignore"):
            # Array API doesn't have reciprocal, use 1/x
            return (1.0 / x,)
