# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy  # type: ignore

from ...mapping import TENSOR_TYPE_TO_NP_TYPE
from ._op_random_common import _CommonRandom


class RandomUniform(_CommonRandom):
    def _run(self, *args):  # type: ignore
        if len(args) != 0:
            raise RuntimeError(
                f"Operator {self.__class__.__name__} cannot have inputs."
            )
        dtype = self._dtype(*args)
        state = self._get_state(self.seed)  # type: ignore
        res = state.rand(*self.shape).astype(dtype)  # type: ignore
        res *= self.high - self.low  # type: ignore
        res += self.low  # type: ignore
        return (res.astype(dtype),)
