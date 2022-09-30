# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from ..op_run import OpRun


class SequenceEmpty(OpRun):
    def _run(self, dtype=None):  # type: ignore
        return ([],)
