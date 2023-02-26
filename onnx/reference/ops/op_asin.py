# SPDX-FileCopyrightText: 2023 ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from onnx.reference.ops._op import OpRunUnaryNum


class Asin(OpRunUnaryNum):
    def _run(self, x):  # type: ignore
        return (np.arcsin(x),)
