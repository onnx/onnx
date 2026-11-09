# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _gemm00(a, b, c, alpha, beta):
    o = np.dot(a, b) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o


def _gemm01(a, b, c, alpha, beta):
    o = np.dot(a, b.T) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o


def _gemm10(a, b, c, alpha, beta):
    o = np.dot(a.T, b) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o


def _gemm11(a, b, c, alpha, beta):
    o = np.dot(a.T, b.T) * alpha
    if c is not None and beta != 0:
        o += c * beta
    return o


class Gemm_6(OpRun):
    def _run(
        self,
        a,
        b,
        c=None,
        alpha=None,
        beta=None,
        transA=None,
        transB=None,
        broadcast=None,
    ):
        if broadcast == 0:
            if transA:
                _meth = _gemm11 if transB else _gemm10
            else:
                _meth = _gemm01 if transB else _gemm00
            res = _meth(a, b, None, alpha, beta)
            if c is None:
                return (res.astype(a.dtype),)
            if c.shape != res.shape:
                raise ValueError(
                    f"Unable to add shape {c.shape} to shape {res.shape} without broadcast."
                )
            return (res + c,)
        if transA:
            _meth = _gemm11 if transB else _gemm10
        else:
            _meth = _gemm01 if transB else _gemm00
        return (_meth(a, b, c, alpha, beta).astype(a.dtype),)


class Gemm_7(OpRun):
    def _run(self, a, b, c=None, alpha=None, beta=None, transA=None, transB=None):
        if transA:
            _meth = _gemm11 if transB else _gemm10
        else:
            _meth = _gemm01 if transB else _gemm00
        return (_meth(a, b, c, alpha, beta).astype(a.dtype),)


class Gemm_29(OpRun):
    def _run(self, a, b, c=None, alpha=None, beta=None, transA=None, transB=None):
        if a.ndim < 1:
            raise ValueError(f"Gemm input A must have rank >= 1, got {a.ndim}.")
        if b.ndim != 2:
            raise ValueError(f"Gemm input B must have rank 2, got {b.ndim}.")
        if transA and a.ndim > 1:
            a = a.swapaxes(-1, -2)
        if transB:
            b = b.T
        result = np.matmul(a, b) * alpha
        if c is not None:
            if c.ndim > result.ndim or any(
                c_dim not in (1, y_dim)
                for c_dim, y_dim in zip(c.shape[::-1], result.shape[::-1], strict=False)
            ):
                raise ValueError(
                    f"Gemm input C shape {c.shape} cannot broadcast to {result.shape}."
                )
            if beta != 0:
                result += c * beta
        return (result.astype(a.dtype),)
