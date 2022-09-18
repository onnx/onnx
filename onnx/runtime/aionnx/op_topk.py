# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...defs import onnx_opset_version
from ..op_run import OpRun


def topk_sorted_implementation(X, k, axis, largest):  # type: ignore
    """
    See function `_kneighbors_reduce_func
    <https://github.com/scikit-learn/scikit-learn/tree/master/
    sklearn/neighbors/base.py#L304>`_.
    """
    if isinstance(k, numpy.ndarray):
        if k.size != 1:
            raise RuntimeError(f"k must be an integer not {k!r}.")
        k = k[0]
    if len(X.shape) == 2 and axis == 1:
        sample_range = numpy.arange(X.shape[0])[:, None]
        if largest == 0:
            sorted_indices = numpy.argpartition(X, axis=axis, kth=k - 1)
            sorted_indices = sorted_indices[:, :k]
            # argpartition doesn't guarantee sorted order, so we sort again
            sorted_indices = sorted_indices[
                sample_range, numpy.argsort(X[sample_range, sorted_indices])
            ]
        else:
            sorted_indices = numpy.argpartition(-X, axis=axis, kth=k - 1)
            sorted_indices = sorted_indices[:, :k]
            # argpartition doesn't guarantee sorted order, so we sort again
            sorted_indices = sorted_indices[
                sample_range, numpy.argsort(-X[sample_range, sorted_indices])
            ]
        sorted_distances = X[sample_range, sorted_indices]
        return sorted_distances, sorted_indices

    sorted_indices = numpy.argsort(X, axis=axis)
    sorted_values = numpy.sort(X, axis=axis)
    if largest:
        sorted_indices = numpy.flip(sorted_indices, axis=axis)
        sorted_values = numpy.flip(sorted_values, axis=axis)
    ark = numpy.arange(k)
    topk_sorted_indices = numpy.take(sorted_indices, ark, axis=axis)
    topk_sorted_values = numpy.take(sorted_values, ark, axis=axis)
    return topk_sorted_values, topk_sorted_indices


class _CommonTopK(OpRun):
    def _common_run(self, data, ink, largest=1):  # type: ignore
        """
        Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what :epkg:`onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        k = ink[0]
        axis = self.axis if self.axis >= 0 else (self.axis + len(data.shape))  # type: ignore
        sort, sorti = topk_sorted_implementation(data, k, axis, largest)
        return (sort, sorti.astype(numpy.int64))


class TopK_1(_CommonTopK):
    def _run(self, data):  # type: ignore
        """
        Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what :epkg:`onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        # TODO: support overridden attributes.
        return _CommonTopK._common_run(self, data, [self.k])  # type: ignore


class TopK_10(_CommonTopK):
    def _run(self, data, ink):  # type: ignore
        """
        Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what :epkg:`onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        return _CommonTopK._common_run(self, data, ink)


class TopK_11(_CommonTopK):
    def __init__(self, onnx_node, run_params):  # type: ignore
        _CommonTopK.__init__(self, onnx_node, run_params)
        if self.sorted not in (True, 1):  # type: ignore
            raise RuntimeError("TopK does not implement anything for sorted=0.")

    def _run(self, data, ink):  # type: ignore
        """
        Runtime for operator *TopK*.
        The implementation is not the most efficient
        as it sorts everything then extracts the top *k*
        values.

        .. warning::
            ONNX specifications may be imprecise in case of negative value
            for axis. The implementation follows what :epkg:`onnxruntime`
            does in `top_k.cc
            <https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/core/providers/cpu/math/top_k.cc#L63>`_.
        """
        # TODO: support overridden attributes.
        return _CommonTopK._common_run(self, data, ink, self.largest)  # type: ignore


if onnx_opset_version() >= 11:
    TopK = TopK_11
elif onnx_opset_version() >= 10:
    TopK = TopK_10  # type: ignore
else:
    TopK = TopK_1  # type: ignore
