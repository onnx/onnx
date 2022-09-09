# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ._op_random_common import _CommonRandom


class RandomNormalLike(_CommonRandom):
    def _run(self, x):  # type: ignore
        dtype = self._dtype(x)
        state = self._get_state(self.seed)  # type: ignore
        res = state.randn(*x.shape).astype(dtype)
        res *= self.scale  # type: ignore
        res += self.mean  # type: ignore
        return (res.astype(dtype),)
