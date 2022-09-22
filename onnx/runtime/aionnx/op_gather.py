# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class Gather(OpRun):
    def _run(self, x, indices):  # type: ignore
        # TODO: support overridden attributes.
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        if not indices.flags["C_CONTIGUOUS"]:
            indices = indices.ascontiguousarray()
        if indices.size == 0:
            return (np.empty((0,), dtype=x.dtype),)
        return (np.take(x, indices, axis=self.axis),)  # type: ignore
