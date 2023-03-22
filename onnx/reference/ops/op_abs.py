# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Abs(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (np.absolute(x),)
