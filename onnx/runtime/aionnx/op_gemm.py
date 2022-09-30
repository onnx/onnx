# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class Gemm(OpRun):
    @staticmethod
    def _gemm00(a, b, c, alpha, beta):  # type: ignore
        o = np.dot(a, b) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm01(a, b, c, alpha, beta):  # type: ignore
        o = np.dot(a, b.T) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm10(a, b, c, alpha, beta):  # type: ignore
        o = np.dot(a.T, b) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    @staticmethod
    def _gemm11(a, b, c, alpha, beta):  # type: ignore
        o = np.dot(a.T, b.T) * alpha
        if c is not None and beta != 0:
            o += c * beta
        return o

    def _run(self, a, b, c=None, alpha=None, beta=None, transA=None, transB=None):  # type: ignore
        if transA:
            _meth = Gemm._gemm11 if transB else Gemm._gemm10
        else:
            _meth = Gemm._gemm01 if transB else Gemm._gemm00
        return (_meth(a, b, c, alpha, beta),)
