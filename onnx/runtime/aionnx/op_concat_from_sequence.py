# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from typing import Any, List

import numpy as np  # type: ignore

from ..op_run import OpRun


def _concat_from_sequence(seq: List[Any], axis: int, new_axis: int = 0) -> np.ndarray:
    if new_axis == 1:
        seq2 = [s[..., np.newaxis] for s in seq]
        res = np.concatenate(seq2, axis=-1)
    else:
        res = np.concatenate(seq, axis=axis)
    return res  # type: ignore


class ConcatFromSequence(OpRun):
    def _run(self, seq):  # type: ignore
        # TODO: support overridden attributes.
        if seq is None:
            raise RuntimeError("A sequence cannot be null.")
        res = _concat_from_sequence(seq, self.axis, new_axis=self.new_axis)  # type: ignore
        return (res,)
