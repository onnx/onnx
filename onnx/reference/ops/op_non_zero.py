# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class NonZero(OpRun):
    def _run(self, x):  # type: ignore
        res = np.vstack(np.nonzero(x))
        return (res,)
