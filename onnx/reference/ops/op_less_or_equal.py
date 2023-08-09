# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
# pylint: disable=W0221

import numpy as np

from onnx.reference.ops._op import OpRunBinaryComparison


class LessOrEqual(OpRunBinaryComparison):
    def _run(self, a, b):  # type: ignore
        return (np.less_equal(a, b),)
