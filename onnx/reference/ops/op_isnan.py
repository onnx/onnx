# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
# pylint: disable=W0221

import numpy as np

from onnx.reference.ops._op import OpRunUnary


class IsNaN(OpRunUnary):
    def _run(self, data):  # type: ignore
        return (np.isnan(data),)
