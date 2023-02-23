# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np

from onnx.reference.ops._op import OpRunBinary


class BitwiseAnd(OpRunBinary):
    def _run(self, x, y):  # type: ignore
        return (np.bitwise_and(x, y),)
