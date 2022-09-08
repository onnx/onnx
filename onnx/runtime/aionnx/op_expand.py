# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


def common_reference_implementation(
    data: numpy.ndarray, shape: numpy.ndarray
) -> numpy.ndarray:
    ones = numpy.ones(shape, dtype=data.dtype)
    return data * ones


class Expand(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, data, shape):  # type: ignore
        return (common_reference_implementation(data, shape),)
