# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnary

# pylint: disable=W0221


class Not(OpRunUnary):
    def _run(self, x):  # type: ignore
        return (np.logical_not(x),)
