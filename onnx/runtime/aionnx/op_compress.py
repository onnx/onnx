# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class Compress(OpRun):
    def _run(self, x, condition):  # type: ignore
        # TODO: support overridden attributes.
        return (np.compress(condition, x, axis=self.axis),)  # type: ignore
