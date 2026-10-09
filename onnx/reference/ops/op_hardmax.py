# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Hardmax(OpRunUnaryNum):
    def _run(self, x, axis=None):
        axis = axis or self.axis
        if x.size == 0:
            return (x,)
        shape = x.shape
        legacy = self.run_params["opsets"][""] < 13
        if legacy:
            if not any(att.name == "axis" for att in self.onnx_node.attribute):
                axis = 1
            axis %= x.ndim
            x = x.reshape((int(np.prod(shape[:axis])), -1))
            axis = 1
        x_argmax = np.argmax(x, axis=axis)
        y = np.zeros_like(x)
        np.put_along_axis(
            y,
            np.expand_dims(x_argmax, axis=axis),
            1,
            axis=axis,
        )
        return (y.reshape(shape),)
