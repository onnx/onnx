# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from ..op_run import OpRun


class CastLike(OpRun):
    def _run(self, x, y):  # type: ignore
        return (x.astype(y.dtype),)
