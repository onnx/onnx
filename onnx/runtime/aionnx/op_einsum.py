# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Einsum(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if not isinstance(self.equation, (str, bytes)):  # type: ignore
            raise TypeError(f"equation must be string but is {type(self.equation)!r}.")  # type: ignore
        self.equation = self.equation.strip()  # type: ignore
        if len(self.equation) == 0:  # type: ignore
            raise TypeError("equation is empty.")

    def _run(self, *args):  # type: ignore
        try:
            return (numpy.einsum(self.equation, *args, optimize=True),)  # type: ignore
        except TypeError:
            return (numpy.einsum(self.equation, *args),)  # type: ignore
