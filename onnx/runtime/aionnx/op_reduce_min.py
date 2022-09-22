# SPDX-License-Identifier: Apache-2.0
# pylint: disable=E1123,W0221

import numpy as np  # type: ignore

from ._op import OpRunReduceNumpy


class ReduceMin(OpRunReduceNumpy):
    def _run(self, data):  # type: ignore
        # TODO: support overridden attributes.
        axes = tuple(self.axes) if self.axes else None
        return (
            np.minimum.reduce(
                data, axis=axes, keepdims=self.keepdims == 1  # type: ignore
            ),
        )
