# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def topk_sorted_implementation(X, k, axis, largest):  # type: ignore
    """See function `_kneighbors_reduce_func
    <https://github.com/scikit-learn/scikit-learn/blob/main/
    sklearn/neighbors/_base.py#L304>`_.
    """
    if isinstance(k, np.ndarray):
        if k.size != 1:
            raise RuntimeError(f"k must be an integer not {k!r}.")
        k = k[0]
    # This conversion is needed for distribution x86.
    k = int(k)
    # Used to tiebreak
    ind_axis = np.indices(X.shape)[axis]
    if largest:
        ind_axis = -ind_axis
    sorted_indices = np.lexsort((ind_axis, X), axis=axis)
    sorted_values = np.take_along_axis(X, sorted_indices, axis=axis)
    if largest:
        sorted_indices = np.flip(sorted_indices, axis=axis)
        sorted_values = np.flip(sorted_values, axis=axis)
    ark = np.arange(k)
    topk_sorted_indices = np.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = np.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices


class _CommonTopK(OpRun):
    def _common_run(self, data, ink, axis, largest=1):  # type: ignore
        """Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what `onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        k = ink[0]
        axis = axis if axis >= 0 else (axis + len(data.shape))  # type: ignore
        sort, sorti = topk_sorted_implementation(data, k, axis, largest)
        return (sort, sorti.astype(np.int64))


class TopK_1(_CommonTopK):
    def _run(self, data, k=None, axis=None):  # type: ignore
        """Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what `onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        return _CommonTopK._common_run(self, data, [k], axis=axis)  # type: ignore


class TopK_10(_CommonTopK):
    def _run(self, data, ink, axis=None):  # type: ignore
        """Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what `onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        return _CommonTopK._common_run(self, data, ink, axis=axis)


class TopK_11(_CommonTopK):
    def _run(
        self,
        data,
        ink,
        axis=None,
        largest=None,
        sorted=None,  # noqa: A002
    ):  # type: ignore
        """Runtime for operator *TopK*.

        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what `onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        if sorted not in (True, 1):
            raise RuntimeError("TopK does not implement anything for sorted=0.")
        return _CommonTopK._common_run(self, data, ink, axis=axis, largest=largest)
