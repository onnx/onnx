# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ...onnx_pb import TensorProto
from ..op_run import OpRun


class EyeLike(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if self.dtype is None:  # type: ignore
            self._dtype = np.float32  # type: ignore
        elif self.dtype == TensorProto.STRING:  # type: ignore
            self._dtype = np.str_  # type: ignore
        else:
            self._dtype = TENSOR_TYPE_TO_NP_TYPE[self.dtype]  # type: ignore

    def _run(self, data, *args):  # type: ignore
        # TODO: support overridden attributes.
        shape = data.shape
        if len(shape) == 1:
            sh = (shape[0], shape[0])
        elif len(shape) == 2:
            sh = shape
        else:
            raise RuntimeError(f"EyeLike only accept 1D or 2D tensors not {shape!r}.")
        return (np.eye(*sh, k=self.k, dtype=self._dtype),)  # type: ignore
