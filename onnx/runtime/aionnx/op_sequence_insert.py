# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

from typing import Any, List

import numpy as np  # type: ignore

from ..op_run import OpRun


def sequence_insert_reference_implementation(
    sequence: List[Any], tensor: np.ndarray, position: np.ndarray = None
) -> List[Any]:
    # make a copy of input sequence
    if sequence is None:
        seq = []
    else:
        seq = list(sequence)
    if position is not None:
        # In these cases, insert_position will be between [-len(sequence), len(sequence)]
        # The position argument will be in the format np.array([pos_index])
        insert_position = position[0]
        seq.insert(insert_position, tensor)
    else:
        # Default position of insertion is at the end of the sequence.
        seq.append(tensor)
    return seq


class SequenceInsert(OpRun):
    def _run(self, S, T, ind=None):  # type: ignore
        if ind is not None and len(ind) > 0:
            res = sequence_insert_reference_implementation(S, T, ind)
        else:
            res = sequence_insert_reference_implementation(S, T)
        return (res,)
