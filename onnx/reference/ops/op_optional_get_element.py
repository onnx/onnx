# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.op_run import OpRun


class OptionalGetElement(OpRun):
    def _run(self, x):  # type: ignore
        if x is None:
            raise ValueError("The requested optional input has no value.")
        return (x,)
