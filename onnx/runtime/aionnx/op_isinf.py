# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ._op import OpRunUnary


class IsInf(OpRunUnary):
    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        if self.detect_negative:  # type: ignore
            if self.detect_positive:  # type: ignore
                return (np.isinf(data),)
            return (np.isneginf(data),)
        if self.detect_positive:  # type: ignore
            return (np.isposinf(data),)
        res = np.full(data.shape, dtype=np.bool_, fill_value=False)
        return (res,)
