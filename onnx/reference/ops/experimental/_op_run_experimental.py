# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun

# pylint: disable=R0913,W0221


class OpRunExperimental(OpRun):
    op_domain = "experimental"
