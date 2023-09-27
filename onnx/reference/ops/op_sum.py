# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class Sum(OpRun):
    def _run(self, *args):  # type: ignore
        return (sum(args).astype(args[0].dtype),)
