# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op import OpRunUnary


class Flatten(OpRunUnary):
    def _run(self, x):  # type: ignore
        # TODO: support overridden attributes.
        i = self.axis  # type: ignore
        shape = x.shape
        new_shape = (1, -1) if i == 0 else (numpy.prod(shape[:i]).astype(int), -1)
        return (x.reshape(new_shape),)  # type: ignore
