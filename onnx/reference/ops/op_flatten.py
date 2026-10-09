# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from math import prod

from onnx.reference.ops._op import OpRunUnary


class Flatten(OpRunUnary):
    def _run(self, x, axis=None):
        i = axis or self.axis
        shape = x.shape
        new_shape = (prod(shape[:i]), prod(shape[i:]))
        return (x.reshape(new_shape),)
