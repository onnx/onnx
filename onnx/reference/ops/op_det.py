# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class Det(OpRun):
    def _run(self, x):  # type: ignore
        res = np.linalg.det(x)
        if not isinstance(res, np.ndarray):
            res = np.array([res])
        return (res,)
