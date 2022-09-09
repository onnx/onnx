# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from typing import Any, List

import numpy  # type: ignore

from ..op_run import OpRun


def _concat_from_sequence(
    seq: List[Any], axis: int, new_axis: int = 0
) -> numpy.ndarray:
    if new_axis == 1:
        seq2 = [s[..., numpy.newaxis] for s in seq]
        res = numpy.concatenate(seq2, axis=-1)
    else:
        res = numpy.concatenate(seq, axis=axis)
    return res


class ConcatFromSequence(OpRun):
    def _run(self, seq):  # type: ignore
        if seq is None:
            raise RuntimeError("A sequence cannot be null.")
        res = _concat_from_sequence(seq, self.axis, new_axis=self.new_axis)  # type: ignore
        return (res,)
