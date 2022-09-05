# SPDX-License-Identifier: Apache-2.0

from ._op import OpRunUnaryNum


class Identity(OpRunUnaryNum):
    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnaryNum.__init__(self, onnx_node, run_params)

    def _run(self, a, attributes=None):  # type: ignore # pylint: disable=W0221
        if a is None:
            return (None,)
        return (a.copy(),)
