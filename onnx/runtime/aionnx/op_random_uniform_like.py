# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from ._op_random_common import _CommonRandom


class RandomUniformLike(_CommonRandom):
    def _run(self, x):  # type: ignore
        dtype = self._dtype(x)
        state = self._get_state()
        res = state.rand(*x.shape).astype(dtype)
        res *= self.high - self.low  # type: ignore
        res += self.low  # type: ignore
        return (res.astype(dtype),)
