# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Trilu(OpRun):
    def _run(self, x, k=None, upper=None):
        self._get_array_api_namespace(x)
        k = 0 if k is None else k.item()
        if upper:
            return (np.triu(x, k),)
        return (np.tril(x, k).astype(x.dtype),)
