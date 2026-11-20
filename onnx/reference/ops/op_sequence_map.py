# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


class SequenceMap(OpRun):
    def _run(self, input_sequence, *additional_inputs, body=None, attributes=None):
        inputs = (input_sequence, *additional_inputs)
        res = [[] for _ in body.output_names]
        for index in range(len(input_sequence)):
            feeds = {
                name: value[index] if isinstance(value, list) else value
                for name, value in zip(body.input_names, inputs, strict=False)
            }
            r = body.run(None, feeds, attributes=attributes)
            for s, i in zip(res, r, strict=False):
                s.append(i)
        return tuple(res)
