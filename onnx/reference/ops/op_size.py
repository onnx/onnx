# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun

# pylint: disable=W0221


class Size(OpRun):
    def _run(self, data):  # type: ignore
        return (np.array(data.size, dtype=np.int64),)
