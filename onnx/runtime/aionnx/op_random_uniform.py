# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ._op_random_common import _CommonRandom


class RandomUniform(_CommonRandom):
    def _run(self):  # type: ignore
        dtype = self._dtype()
        state = self._get_state()
        res = state.rand(*self.shape).astype(dtype)  # type: ignore
        res *= self.high - self.low  # type: ignore
        res += self.low  # type: ignore
        return (res.astype(dtype),)
