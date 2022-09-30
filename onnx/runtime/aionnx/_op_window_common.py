# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0613,W0221

import numpy as np  # type: ignore

from ...helper import tensor_dtype_to_np_dtype
from ..op_run import OpRun


class _CommonWindow(OpRun):
    @staticmethod
    def _begin(size, periodic, output_datatype):  # type: ignore
        dtype = tensor_dtype_to_np_dtype(output_datatype)
        if periodic == 1:
            N_1 = size
        else:
            N_1 = size - 1
        ni = np.arange(size, dtype=dtype)
        return ni, N_1

    @staticmethod
    def _end(size, res, output_datatype):  # type: ignore
        dtype = tensor_dtype_to_np_dtype(output_datatype)
        return (res.astype(dtype),)
