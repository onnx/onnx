# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun

# pylint: disable=W0221,W0613


class SequenceEmpty(OpRun):
    def _run(self, dtype=None):  # type: ignore
        return ([],)
