# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0913,W0221

from onnx.reference.op_run import OpRun


class OpRunAiOnnxIo(OpRun):
    op_domain = "ai.onnx.io"
