# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,W0221

import numpy as np

from onnx.reference.op_run import OpRun


class StringConcat(OpRun):
    def _run(self, x, y):
        return (np.char.add(x, y),)
