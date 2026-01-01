# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from onnx.reference.ops._op import OpRunUnaryNum


class Erf(OpRunUnaryNum):
    def _run(self, x):
        # Array API 2023 includes erf as an extension
        xp = self._get_array_api_namespace(x)
        # Try to use array API erf if available, fallback to numpy
        if hasattr(xp, 'erf'):
            return (xp.erf(x),)
        else:
            # Fallback for array API implementations without erf
            from math import erf
            import numpy as np
            from onnx.reference.array_api_namespace import asarray
            
            erf_vec = np.vectorize(erf, otypes=["f"])
            result = erf_vec(x)
            if xp.__name__ != 'numpy' and 'numpy' not in str(xp.__name__):
                result = asarray(result, xp=xp)
            return (result,)
