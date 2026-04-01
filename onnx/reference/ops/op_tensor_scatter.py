# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class TensorScatter(OpRun):
    def _run(self, past_cache, update, write_indices=None, mode="linear", axis=-2):
        if mode not in {"linear", "circular"}:
            raise ValueError(f"Unsupported mode: {mode}")

        if write_indices is None:
            write_indices = np.zeros((past_cache.shape[0],), dtype=np.int64)

        input_shape = past_cache.shape
        update_shape = update.shape
        axis = axis % len(input_shape)

        for i in range(len(input_shape)):
            if i != axis:
                if input_shape[i] != update_shape[i]:
                    raise ValueError(
                        f"Input shape {input_shape} and update shape {update_shape} are not compatible in dimension {i}"
                    )
            if i == axis:
                if input_shape[i] < update_shape[i]:
                    raise ValueError(
                        f"Input shape {input_shape} and update shape {update_shape} are not compatible in axis dimension"
                    )

        max_sequence_length = input_shape[axis]
        sequence_length = update_shape[axis]
        present_cache = np.copy(past_cache)

        for prefix_idx in np.ndindex(input_shape[:axis]):
            batch_idx = prefix_idx[0]
            for sequence_idx in range(sequence_length):
                cache_idx = (*prefix_idx, write_indices[batch_idx] + sequence_idx)
                if mode == "circular":
                    cache_idx = tuple(
                        np.mod(np.asarray(cache_idx), max_sequence_length)
                    )
                update_idx = (*prefix_idx, sequence_idx)
                present_cache[cache_idx] = update[update_idx]

        return (present_cache.reshape(input_shape),)
