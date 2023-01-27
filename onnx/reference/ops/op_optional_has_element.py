# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class OptionalHasElement(OpRun):
    def _run(self, x=None):  # type: ignore
        if x is None:
            return ([],)
        if isinstance(x, list):
            (e,) = x
            return (np.array(e is not None),)
        elif isinstance(x, np.ndarray):
            return (np.array(True),)
        return ([],)
