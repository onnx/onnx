# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Softsign(OpRunUnaryNum):
    def _run(self, X):
        xp = self._get_array_api_namespace(X)
        tmp = np.abs(X)
        tmp += 1
        np.divide(X, tmp, out=tmp)
        return (tmp,)