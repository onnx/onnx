# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun
from onnx.reference.ops._op_common_threefry import threefry_seed


class InitPRNG(OpRun):
    def _run(self, seed):
        return (threefry_seed(seed),)
