# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnary

# pylint: disable=W0221


class IsNaN(OpRunUnary):
    def _run(self, data):  # type: ignore
        return (np.isnan(data),)
