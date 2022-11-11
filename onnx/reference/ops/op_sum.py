# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from onnx.reference.op_run import OpRun


class Sum(OpRun):
    def _run(self, *args):  # type: ignore
        return (sum(args),)
