# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun

# pylint: disable=W0221


class SequenceAt(OpRun):
    def _run(self, seq, index):  # type: ignore
        return (seq[index],)
