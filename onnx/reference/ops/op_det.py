# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun

# pylint: disable=W0221


class Det(OpRun):
    def _run(self, x):  # type: ignore
        return (np.linalg.det(x),)
