# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunBinaryNumpy


class Div(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNumpy.__init__(self, numpy.divide, onnx_node, run_params)

    def _run(self, a, b, attributes=None, verbose=0, fLOG=None):  # type: ignore
        res = OpRunBinaryNumpy._run(self, a, b)
        if res[0].dtype != a.dtype:
            return (res[0].astype(a.dtype),)
        return res
