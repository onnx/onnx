# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class AttributeHasValue(OpRun):
    def _run(self, *args, **kwargs):  # noqa: ARG002
        # TODO: support overridden attributes.
        for att in self.onnx_node.attribute:
            if att.name.startswith("value_"):
                return (np.array([True]),)
        return (np.array([False]),)
