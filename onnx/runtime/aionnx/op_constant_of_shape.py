# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class ConstantOfShape(OpRun):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        self.cst = (
            self.value[0] if isinstance(self.value, numpy.ndarray) else self.value  # type: ignore
        )
        if isinstance(self.cst, int):
            self.cst = numpy.int64(self.cst)
        elif isinstance(self.cst, float):
            self.cst = numpy.float64(self.cst)
        if not isinstance(
            self.cst,
            (
                numpy.float32,
                numpy.float64,
                numpy.int64,
                numpy.int32,
                numpy.bool_,
                numpy.float16,
            ),
        ):
            raise TypeError(f"cst must be a real not {type(self.cst)}")

    def _run(self, data):  # type: ignore
        try:
            res = numpy.full(tuple(data), self.cst)  # type: ignore
        except TypeError as e:
            raise RuntimeError(
                f"Unable to create a constant of shape {data!r} with value {self.cst!r} "  # type: ignore
                f"(raw value={self.value!r})."  # type: ignore
            ) from e
        return (res,)
