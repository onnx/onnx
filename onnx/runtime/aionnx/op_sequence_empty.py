# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class SequenceEmpty(OpRun):
    def _run(self):  # type: ignore
        return ([],)
