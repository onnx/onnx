# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinary

# pylint: disable=W0221


class Xor(OpRunBinary):
    def _run(self, x, y):  # type: ignore
        return (np.logical_xor(x, y),)
