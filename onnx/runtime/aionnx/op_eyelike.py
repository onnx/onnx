# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ...helper import tensor_dtype_to_np_dtype
from ...onnx_pb import TensorProto
from ..op_run import OpRun


class EyeLike(OpRun):
    def _run(self, data, *args, dtype=None, k=None):  # type: ignore
        if dtype is None:
            _dtype = np.float32
        elif dtype == TensorProto.STRING:
            _dtype = np.str_
        else:
            _dtype = tensor_dtype_to_np_dtype(dtype)
        shape = data.shape
        if len(shape) == 1:
            sh = (shape[0], shape[0])
        elif len(shape) == 2:
            sh = shape
        else:
            raise RuntimeError(f"EyeLike only accept 1D or 2D tensors not {shape!r}.")
        return (np.eye(*sh, k=k, dtype=_dtype),)  # type: ignore
