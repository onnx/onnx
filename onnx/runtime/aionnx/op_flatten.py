# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnary


class Flatten(OpRunUnary):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnary.__init__(self, onnx_node, run_params)

    def _run(self, x):  # type: ignore
        i = self.axis  # type: ignore
        shape = x.shape
        new_shape = (1, -1) if i == 0 else (numpy.prod(shape[:i]).astype(int), -1)
        return (x.reshape(new_shape),)  # type: ignore
