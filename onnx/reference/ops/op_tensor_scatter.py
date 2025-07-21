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

        print(input_shape, update_shape, axis)
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

        batch_size = input_shape[0]
        max_seq_len = input_shape[axis]
        seq_len = update.shape[axis]

        present_cache = np.copy(past_cache)

        # Reshape from (batch_size, D1, D2, ..., Dn) to (batch_size, D1 * ... * D(axis-1), D(axis), D(axis+1) * ... * Dn)
        new_shape = (
            batch_size,
            np.prod(input_shape[1:axis], dtype=np.int64),
            -1,
            np.prod(input_shape[axis + 1 :], dtype=np.int64),
        )
        present_cache = present_cache.reshape(new_shape)
        update = update.reshape(new_shape)

        for i in range(batch_size):
            start_idx = write_indices[i]
            for j in range(new_shape[1]):
                for k in range(seq_len):
                    idx = start_idx + k
                    if mode == "circular":
                        idx = idx % max_seq_len
                    present_cache[i, j, idx, :] = update[i, j, k, :]

        return (present_cache.reshape(input_shape),)
