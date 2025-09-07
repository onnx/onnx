# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.helper import np_dtype_to_tensor_dtype
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_cast import cast_to


def _cast_like(x, y, saturate: bool):
    return (cast_to(x, np_dtype_to_tensor_dtype(y.dtype), saturate),)


class CastLike_15(OpRun):
    def _run(self, x, y):
        return _cast_like(x, y, True)


class CastLike_19(OpRun):
    def _run(self, x, y, saturate=False):
        return _cast_like(x, y, saturate)
