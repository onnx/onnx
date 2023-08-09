# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnary

# pylint: disable=W0221


class BitwiseNot(OpRunUnary):
    def _run(self, X):
        return (np.bitwise_not(X),)
