# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from onnx.reference.op_run import OpRun


class OptionalGetElement_15(OpRun):
    def _run(self, x):  # type: ignore
        if x is None:
            raise ValueError("The requested optional input has no value.")
        return (x,)

class OptionalGetElement_18(OpRun):
    def _run(self, x, type=None):  # type: ignore
        if x is None:
            raise ValueError("The requested optional input has no value.")
        return (x,)
