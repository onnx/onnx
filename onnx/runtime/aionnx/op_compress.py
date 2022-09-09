# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ..op_run import OpRun


class Compress(OpRun):
    def _run(self, x, condition):  # type: ignore
        return (numpy.compress(condition, x, axis=self.axis),)  # type: ignore
