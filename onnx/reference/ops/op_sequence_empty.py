# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
# pylint: disable=W0221,W0613

from onnx.reference.op_run import OpRun


class SequenceEmpty(OpRun):
    def _run(self, dtype=None):  # type: ignore
        return ([],)
