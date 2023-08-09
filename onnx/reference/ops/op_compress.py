# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class Compress(OpRun):
    def _run(self, x, condition, axis=None):  # type: ignore
        return (np.compress(condition, x, axis=axis),)  # type: ignore
