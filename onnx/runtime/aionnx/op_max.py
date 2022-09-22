# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunBinaryNumpy


class Max(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinarynp.__init__(self, np.maximum, onnx_node, run_params)

    def run(self, *data):  # type: ignore
        if len(data) == 2:
            return OpRunBinarynp.run(self, *data)
        if len(data) == 1:
            return (data[0].copy(),)
        if len(data) > 2:
            a = data[0]
            for i in range(1, len(data)):
                a = np.maximum(a, data[i])
            return (a,)
        raise RuntimeError("Unexpected turn of events.")
