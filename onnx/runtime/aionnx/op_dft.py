# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from typing import Sequence

import numpy  # type: ignore

from ..op_run import OpRun


def _fft(x: numpy.ndarray, fft_length: Sequence[int], axis: int) -> numpy.ndarray:
    if fft_length is None:
        fft_length = [x.shape[axis]]
    ft = numpy.fft.fft(x, fft_length[0], axis=axis)
    r = numpy.real(ft)
    i = numpy.imag(ft)
    merged = numpy.vstack([r[numpy.newaxis, ...], i[numpy.newaxis, ...]])
    perm = numpy.arange(len(merged.shape))
    perm[:-1] = perm[1:]
    perm[-1] = 0
    tr = numpy.transpose(merged, list(perm))
    if tr.shape[-1] != 2:
        raise RuntimeError(
            f"Unexpected shape {tr.shape}, x.shape={x.shape} "
            f"fft_length={fft_length}."
        )
    return tr


def _cfft(
    x: numpy.ndarray,
    fft_length: Sequence[int],
    axis: int,
    onesided: bool = False,
    normalize: bool = False,
) -> numpy.ndarray:
    if normalize:
        raise RuntimeError("DFT is not implemented when normalize is True.")
    if x.shape[-1] == 1:
        tmp = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        tmp = real + 1j * imag
    c = numpy.squeeze(tmp, -1)
    res = _fft(c, fft_length, axis=axis)
    if onesided:
        slices = [slice(0, a) for a in res.shape]
        slices[axis] = slice(0, res.shape[axis] // 2 + 1)
        return res[tuple(slices)]
    return res


def _ifft(
    x: numpy.ndarray, fft_length: Sequence[int], axis: int = -1, onesided: bool = False
) -> numpy.ndarray:
    ft = numpy.fft.ifft(x, fft_length[0], axis=axis)
    r = numpy.real(ft)
    i = numpy.imag(ft)
    merged = numpy.vstack([r[numpy.newaxis, ...], i[numpy.newaxis, ...]])
    perm = numpy.arange(len(merged.shape))
    perm[:-1] = perm[1:]
    perm[-1] = 0
    tr = numpy.transpose(merged, list(perm))
    if tr.shape[-1] != 2:
        raise RuntimeError(
            f"Unexpected shape {tr.shape}, x.shape={x.shape} "
            f"fft_length={fft_length}."
        )
    if onesided:
        slices = [slice(a) for a in tr.shape]
        slices[axis] = slice(0, tr.shape[axis] // 2 + 1)
        return tr[tuple(slices)]
    return tr


def _cifft(
    x: numpy.ndarray, fft_length: Sequence[int], axis: int = -1, onesided: bool = False
) -> numpy.ndarray:
    if x.shape[-1] == 1:
        tmp = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        tmp = real + 1j * imag
    c = numpy.squeeze(tmp, -1)
    return _ifft(c, fft_length, axis=axis, onesided=onesided)


class DFT(OpRun):
    def _run(self, x, dft_length=None):  # type: ignore
        if dft_length is None:
            dft_length = numpy.array([x.shape[self.axis]], dtype=numpy.int64)  # type: ignore
        if self.inverse:  # type: ignore
            res = _cifft(x, dft_length, axis=self.axis, onesided=self.onesided)  # type: ignore
        else:
            res = _cfft(x, dft_length, axis=self.axis, onesided=self.onesided)  # type: ignore
        return (res.astype(x.dtype),)
