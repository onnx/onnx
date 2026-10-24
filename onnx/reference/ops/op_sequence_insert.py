# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numpy as np

from onnx.reference.op_run import OpRun


def sequence_insert_reference_implementation(
    sequence: list[Any] | np.ndarray,
    tensor: np.ndarray,
    position: np.ndarray | None = None,
) -> list[Any]:
    # make a copy of input sequence
    seq: list[Any] = []
    if sequence is not None and (
        not isinstance(sequence, np.ndarray) or len(sequence.shape) > 0
    ):
        try:
            seq.extend(sequence)
        except TypeError as e:
            raise TypeError(
                f"Unable to iterate on type {type(sequence)}: {sequence}."
            ) from e
    if position is not None:
        # The position argument will be in the format np.array([pos_index]).
        # Accepted positions are in [-len(seq), len(seq)]; list.insert counts
        # negative positions from the back and appends when the position
        # equals len(seq), as the specification requires.
        insert_position = int(position[0])
        if not -len(seq) <= insert_position <= len(seq):
            raise ValueError(
                f"Position {insert_position} is out of range [{-len(seq)}, {len(seq)}]."
            )
        seq.insert(insert_position, tensor)
    else:
        # Default position of insertion is at the end of the sequence.
        seq.append(tensor)
    return seq


class SequenceInsert(OpRun):
    def _run(self, S, T, ind=None):
        if ind is None:
            res = sequence_insert_reference_implementation(S, T)
        elif isinstance(ind, int):
            res = sequence_insert_reference_implementation(S, T, [ind])  # type: ignore[arg-type]
        elif len(ind.shape) > 0:
            res = sequence_insert_reference_implementation(S, T, ind)
        elif len(ind.shape) == 0:
            res = sequence_insert_reference_implementation(S, T, [int(ind)])  # type: ignore[arg-type]
        else:
            res = sequence_insert_reference_implementation(S, T)
        return (res,)
