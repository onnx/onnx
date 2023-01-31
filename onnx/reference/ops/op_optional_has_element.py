# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class OptionalHasElement(OpRun):
    def _run(self, x=None):  # type: ignore
        if x is None:
            return (np.array(False),)
        if isinstance(x, (np.ndarray, list)):
            return (np.array(True),)
        return (np.array(False),)
