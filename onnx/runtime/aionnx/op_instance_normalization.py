# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0914,W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class InstanceNormalization(OpRun):
    def _run(self, x, s, bias):  # type: ignore
        dims_x = len(x.shape)
        axis = tuple(range(2, dims_x))
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        dim_ones = (1,) * (dims_x - 2)
        s = s.reshape(-1, *dim_ones)
        bias = bias.reshape(-1, *dim_ones)
        y = s * (x - mean) / np.sqrt(var + self.epsilon) + bias  # type: ignore
        return (y.astype(x.dtype),)
