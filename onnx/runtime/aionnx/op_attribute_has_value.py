# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class AttributeHasValue(OpRun):
    def _run(  # type: ignore
        self,
    ):
        for att in self.onnx_node.attribute:
            if att.name.startswith("value_"):
                return (np.array([True]),)
        return (np.array([False]),)
