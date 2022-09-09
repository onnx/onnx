# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ..op_run import OpRun


class _CommonRandom(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if len(self.shape) == 0:  # type: ignore
            raise ValueError(  # pragma: no cover
                f"shape cannot be empty for operator {self.__class__.__name__}."
            )
        self.numpy_type = TENSOR_TYPE_TO_NP_TYPE[self.dtype]  # type: ignore

    def _dtype(self, *data, dtype_first=False):  # type: ignore
        if dtype_first:
            if self.dtype != 0:  # type: ignore
                return self.numpy_type
            if len(data) > 0:
                return data[0].dtype
            raise RuntimeError(
                f"dtype cannot be None for operator {self.__class__.__name__!r}, "
                f"self.numpy_type={self.numpy_type}, len(data)={len(data)}."
            )
        res = None
        if len(data) == 0:
            res = self.numpy_type
        elif self.numpy_type is not None:
            res = self.numpy_type
        elif hasattr(data[0], "dtype"):
            res = data[0].dtype
        if res is None:
            raise RuntimeError(
                f"dtype cannot be None for operator {self.__class__.__name__!r}, "
                f"self.numpy_type={self.numpy_type}, type(data[0])={type(data[0])}."
            )
        return res

    def _get_state(self, seed):  # type: ignore
        if numpy.isnan(self.seed):  # type: ignore
            state = numpy.random.RandomState()
        else:
            state = numpy.random.RandomState(seed=int(self.seed))  # type: ignore
        return state
