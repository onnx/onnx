# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class OptionalGetElement(OpRun):
    def _run(self, x):  # type: ignore
        if x is None:
            return ([],)
        if isinstance(x, list):
            return (x,)
        if isinstance(x, np.ndarray):
            return (x,)
        raise RuntimeError("Input is empty.")
