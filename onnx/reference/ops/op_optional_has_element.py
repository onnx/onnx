# SPDX-FileCopyrightText: 2023 ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from onnx.reference.op_run import OpRun


class OptionalHasElement(OpRun):
    def _run(self, x=None):  # type: ignore
        return (np.array(x is not None),)
