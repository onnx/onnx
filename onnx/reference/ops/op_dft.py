# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import numpy as np

from onnx.reference.op_run import OpRun


def _fft(x: np.ndarray, fft_length: Sequence[int], axis: int) -> np.ndarray:
    assert fft_length is not None

    transformed = np.fft.fft(x, fft_length[0], axis=axis)
    real_frequencies = np.real(transformed)
    imaginary_frequencies = np.imag(transformed)
    # TODO(justinchuby): Just concat on the last axis and remove transpose
    merged = np.vstack(
        [real_frequencies[np.newaxis, ...], imaginary_frequencies[np.newaxis, ...]]
    )
    perm = np.arange(len(merged.shape))
    perm[:-1] = perm[1:]
    perm[-1] = 0
    transposed = np.transpose(merged, list(perm))
    if transposed.shape[-1] != 2:
        raise RuntimeError(
            f"Unexpected shape {transposed.shape}, x.shape={x.shape} "
            f"fft_length={fft_length}."
        )
    return transposed


def _cfft(
    x: np.ndarray,
    fft_length: Sequence[int],
    axis: int,
    onesided: bool,
    normalize: bool,
) -> np.ndarray:
    if x.shape[-1] == 1:
        tmp = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        tmp = real + 1j * imag
    complex_signals = np.squeeze(tmp, -1)
    result = _fft(complex_signals, fft_length, axis=axis)
    if onesided:
        slices = [slice(0, a) for a in result.shape]
        slices[axis] = slice(0, result.shape[axis] // 2 + 1)
        result = result[tuple(slices)]  # type: ignore
    if normalize:
        if len(fft_length) == 1:
            result /= fft_length[0]
        else:
            raise NotImplementedError(
                f"normalize=True not implemented for fft_length={fft_length}."
            )
    return result


def _ifft(
    x: np.ndarray, fft_length: Sequence[int], axis: int, onesided: bool
) -> np.ndarray:
    signals = np.fft.ifft(x, fft_length[0], axis=axis)
    real_signals = np.real(signals)
    imaginary_signals = np.imag(signals)
    # TODO(justinchuby): Just concat on the last axis and remove transpose
    merged = np.vstack(
        [real_signals[np.newaxis, ...], imaginary_signals[np.newaxis, ...]]
    )
    perm = np.arange(len(merged.shape))
    perm[:-1] = perm[1:]
    perm[-1] = 0
    transposed = np.transpose(merged, list(perm))
    if transposed.shape[-1] != 2:
        raise RuntimeError(
            f"Unexpected shape {transposed.shape}, x.shape={x.shape} "
            f"fft_length={fft_length}."
        )
    if onesided:
        slices = [slice(a) for a in transposed.shape]
        slices[axis] = slice(0, transposed.shape[axis] // 2 + 1)
        return transposed[tuple(slices)]
    return transposed


def _cifft(
    x: np.ndarray, fft_length: Sequence[int], axis: int = -1, onesided: bool = False
) -> np.ndarray:
    if x.shape[-1] == 1:
        tmp = x
    else:
        slices = [slice(0, x) for x in x.shape]
        slices[-1] = slice(0, x.shape[-1], 2)
        real = x[tuple(slices)]
        slices[-1] = slice(1, x.shape[-1], 2)
        imag = x[tuple(slices)]
        tmp = real + 1j * imag
    complex_signals = np.squeeze(tmp, -1)
    return _ifft(complex_signals, fft_length, axis=axis, onesided=onesided)


class DFT_17(OpRun):
    def _run(self, x, dft_length: Sequence[int] | None = None, axis=1, inverse: bool = False, onesided: bool = False) -> tuple[np.ndarray]:  # type: ignore
        if dft_length is None:
            dft_length = (x.shape[axis],)
        if inverse:  # type: ignore
            res = _cifft(x, dft_length, axis=axis, onesided=onesided)
        else:
            res = _cfft(x, dft_length, axis=axis, onesided=onesided, normalize=False)
        return (res.astype(x.dtype),)


class DFT_20(OpRun):
    def _run(self, x, axis: int = -1, dft_length: Sequence[int] | None = None, inverse: bool = False, onesided: bool = False) -> tuple[np.ndarray]:  # type: ignore
        if dft_length is None:
            dft_length = (x.shape[axis],)
        if inverse:  # type: ignore
            res = _cifft(x, dft_length, axis=axis, onesided=onesided)
        else:
            res = _cfft(x, dft_length, axis=axis, onesided=onesided, normalize=False)
        return (res.astype(x.dtype),)
