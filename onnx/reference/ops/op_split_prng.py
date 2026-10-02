# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_threefry import threefry_fold_in, threefry_split


class SplitPRNG(OpRun):
    def __init__(self, onnx_node, run_params):
        OpRun.__init__(self, onnx_node, run_params)
        self.n_outputs = len(onnx_node.output)

    def _run(self, prng_state, data=None):
        if data is None:
            return threefry_split(prng_state, self.n_outputs)

        data = np.asarray(data)
        if data.dtype != np.int64 or data.shape != (self.n_outputs,):
            raise ValueError(
                "SplitPRNG data must be an int64 vector containing one element "
                f"per output; got {data.dtype} with shape {data.shape}."
            )
        return tuple(threefry_fold_in(prng_state, int(value)) for value in data)
