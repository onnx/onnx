# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Where(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)

    def _run(self, condition, x, y):  # type: ignore
        if x.dtype != y.dtype and x.dtype not in (numpy.object_,):
            raise RuntimeError(
                f"x and y should share the same dtype {x.dtype} != {y.dtype}"
            )
        return (numpy.where(condition, x, y).astype(x.dtype),)
