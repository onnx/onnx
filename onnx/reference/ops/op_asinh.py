# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
# pylint: disable=W0221

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Asinh(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (np.arcsinh(x),)
