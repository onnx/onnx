# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Size(OpRun):
    def _run(self, data):  # type: ignore
        return (numpy.array(data.size, dtype=numpy.int64),)
