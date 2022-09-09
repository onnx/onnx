# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ._op_random_common import _CommonRandom


class Bernoulli(_CommonRandom):
    def _run(self, x):  # type: ignore
        dtype = self._dtype(x, dtype_first=True)
        state = self._get_state(self.seed)  # type: ignore
        res = state.binomial(1, p=x).astype(dtype)
        return (res.astype(dtype),)
