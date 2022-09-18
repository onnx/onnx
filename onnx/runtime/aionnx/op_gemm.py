# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Gemm(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if self.transA:  # type: ignore
            _meth = Gemm._gemm11 if self.transB else Gemm._gemm10  # type: ignore
        else:
            _meth = Gemm._gemm01 if self.transB else Gemm._gemm00  # type: ignore
        self._meth = lambda a, b, c: _meth(a, b, c, self.alpha, self.beta)  # type: ignore

    @staticmethod
    def _gemm00(a, b, c, alpha, beta):  # type: ignore
        o = numpy.dot(a, b) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm01(a, b, c, alpha, beta):  # type: ignore
        o = numpy.dot(a, b.T) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm10(a, b, c, alpha, beta):  # type: ignore
        o = numpy.dot(a.T, b) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm11(a, b, c, alpha, beta):  # type: ignore
        o = numpy.dot(a.T, b.T) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    def _run(self, a, b, c=None):  # type: ignore
        # TODO: support overridden attributes.
        return (self._meth(a, b, c),)
