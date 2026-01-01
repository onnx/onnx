# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops._op import OpRunBinaryNumpy


class Max(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):
        OpRunBinaryNumpy.__init__(self, np.maximum, onnx_node, run_params)

    def run(self, *data):
        if len(data) == 2:
            return OpRunBinaryNumpy.run(self, *data)
        if len(data) == 1:
            return (data[0].copy(),)
        if len(data) > 2:
            xp = self._get_array_api_namespace(*data)
            a = data[0]
            for i in range(1, len(data)):
                a = xp.maximum(a, data[i])
            return (a,)
        raise RuntimeError("Unexpected turn of events.")
