# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class ReverseSequence(OpRun):
    def _run(self, data, sequence_lens, batch_axis=None, time_axis=None):
        index = [slice(0, s) for s in data.shape]
        index_data = [slice(0, s) for s in data.shape]
        result = data.copy()
        for i, sl in enumerate(sequence_lens):
            index[batch_axis] = i
            index[time_axis] = slice(0, sl)
            index_data[batch_axis] = i
            index_data[time_axis] = slice(sl - 1, None, -1)
            result[tuple(index)] = data[tuple(index_data)]
        return (result,)
