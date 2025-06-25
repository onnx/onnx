# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.reference.op_run import OpRun


def cast_to(x, to, saturate: bool):
    # TODO(justinchuby): Implement support for saturate=False
    if not saturate:
        raise NotImplementedError("saturate=False is not supported yet.")
    if to == onnx.TensorProto.STRING:
        return x.astype(np.str_)

    dtype = onnx.helper.tensor_dtype_to_np_dtype(to)
    return x.astype(dtype)


class Cast_1(OpRun):
    def _run(self, x, to=None):
        return (cast_to(x, to, saturate=True),)


class Cast_19(OpRun):
    def _run(self, x, to=None, saturate: bool = True):
        return (cast_to(x, to, saturate),)
