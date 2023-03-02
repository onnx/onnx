# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class NonZero(OpRun):
    def _run(self, x):  # type: ignore
        # Specify np.int64 for Windows x86 machines
        res = np.array(np.vstack(np.nonzero(x)), dtype=np.int64)
        return (res,)
