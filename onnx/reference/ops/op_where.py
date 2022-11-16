# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class Where(OpRun):
    def _run(self, condition, x, y):  # type: ignore
        if x.dtype != y.dtype and x.dtype not in (np.object_,):
            raise RuntimeError(
                f"x and y should share the same dtype {x.dtype} != {y.dtype}"
            )
        return (np.where(condition, x, y).astype(x.dtype),)
