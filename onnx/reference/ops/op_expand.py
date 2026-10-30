# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def common_reference_implementation(data: np.ndarray, shape: np.ndarray) -> np.ndarray:
    output_shape = np.broadcast_shapes(data.shape, tuple(shape))
    return np.broadcast_to(data, output_shape).copy()


class Expand(OpRun):
    def _run(self, data, shape):
        return (common_reference_implementation(data, shape),)
