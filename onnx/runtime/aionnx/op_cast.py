# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ...onnx_pb import TensorProto
from ..op_run import OpRun


class Cast(OpRun):  # type: ignore
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRun.__init__(self, onnx_node, run_params)
        if self.to == TensorProto.STRING:  # type: ignore
            self._dtype = numpy.str_
        else:
            self._dtype = TENSOR_TYPE_TO_NP_TYPE[self.to]  # type: ignore
        self._cast = lambda x: x.astype(self._dtype)

    def _run(self, x):  # type: ignore
        return (self._cast(x),)
