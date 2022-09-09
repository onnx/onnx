# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnaryNum


class Transpose(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)
        self.perm_ = None if (self.perm is None or len(self.perm) == 0) else self.perm  # type: ignore

    def _run(self, data):  # type: ignore
        if self.perm_ is None:
            return (numpy.transpose(data),)
        if len(self.perm_) != len(data.shape):
            raise RuntimeError(
                f"Inconsistent permutation {self.perm_!r} with shape {data.shape!r}."
            )
        return (numpy.transpose(data, axes=self.perm_),)
