# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class OptionalHasElement(OpRun):
    def _run(self, x=None):  # type: ignore
        if x is None:
            return ([],)
        if isinstance(x, list):
            if len(x) > 0:
                return (np.array([e is not None for e in x]),)
        elif isinstance(x, np.ndarray):
            if len(x.shape) > 0 and x.shape[0] > 0:
                return (np.array([e is not None for e in x]),)
        return ([],)
