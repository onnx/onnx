# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.ops._op import OpRunBinaryNumpy

# pylint: disable=W0221


class Add(OpRunBinaryNumpy):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunBinaryNumpy.__init__(self, np.add, onnx_node, run_params)
