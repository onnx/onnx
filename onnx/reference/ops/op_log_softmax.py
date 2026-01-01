# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops.op_softmax import Softmax


class LogSoftmax(Softmax):
    def _run(self, X):
        xp = self._get_array_api_namespace(X)
        Y = Softmax._run(self, X)[0]
        return (xp.log(Y),)
