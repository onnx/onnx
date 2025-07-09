# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class TensorScatter(OpRun):
    def _run(self, past_cache, update, write_indices=None, mode="linear"):
        if write_indices is None:
            write_indices = np.zeros((past_cache.shape[0],), dtype=np.int64)

        present_cache = np.copy(past_cache)
        batch_size = past_cache.shape[0]
        max_seq_len = past_cache.shape[-2]
        seq_len = update.shape[-2]

        if mode == "linear":
            for i in range(batch_size):
                start = write_indices[i]
                end = start + seq_len
                present_cache[i, :, start:end, :] = update[i]
        elif mode == "circular":
            for i in range(batch_size):
                for j in range(seq_len):
                    index = (write_indices[i] + j) % max_seq_len
                    present_cache[i, :, index, :] = update[i, :, j, :]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return (present_cache,)


