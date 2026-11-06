# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunUnaryNum
from onnx.reference.ops._op_common_softmax import _log_softmax


class LogSoftmax(OpRunUnaryNum):
    def _run(self, X):
        return (_log_softmax(X, self.axis),)
