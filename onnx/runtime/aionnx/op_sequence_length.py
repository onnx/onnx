# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class SequenceLength(OpRun):
    def _run(self, input_sequence):  # type: ignore
        return (numpy.array(len(input_sequence)),)
