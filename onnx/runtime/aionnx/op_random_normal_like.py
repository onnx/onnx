# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ._op_random_common import _CommonRandom


class RandomNormalLike(_CommonRandom):
    def _run(self, x):  # type: ignore
        dtype = self._dtype(x)
        state = self._get_state()
        res = state.randn(*x.shape).astype(dtype)
        res *= self.scale  # type: ignore
        res += self.mean  # type: ignore
        return (res.astype(dtype),)
