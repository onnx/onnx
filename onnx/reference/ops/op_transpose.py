# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class Transpose(OpRun):
    def _run(self, data, perm=None):
        xp = self._get_array_api_namespace(data)
        perm_ = None if (perm is None or len(perm) == 0) else perm
        if perm_ is None:
            # Array API uses axes instead of perm
            return (xp.permute_dims(data, list(reversed(range(len(data.shape))))),)
        if len(perm_) != len(data.shape):
            raise RuntimeError(
                f"Inconsistent permutation {perm_!r} with shape {data.shape!r}."
            )
        return (xp.permute_dims(data, perm_),)
