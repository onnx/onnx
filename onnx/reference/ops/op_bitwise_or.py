# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
# pylint: disable=W0221

import numpy as np

from onnx.reference.ops._op import OpRunBinary


class BitwiseOr(OpRunBinary):
    def _run(self, x, y):  # type: ignore
        return (np.bitwise_or(x, y),)
