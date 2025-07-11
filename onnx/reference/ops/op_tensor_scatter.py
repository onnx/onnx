# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


class TensorScatter(OpRun):
    def _run(self, past_cache, update, write_indices=None, mode="linear"):
        if mode not in ["linear", "circular"]:
            raise ValueError(f"Unsupported mode: {mode}")

        if write_indices is None:
            write_indices = np.zeros((past_cache.shape[0],), dtype=np.int64)

        present_cache = np.copy(past_cache)
        batch_size = past_cache.shape[0]
        
        if past_cache.ndim == 3:
            max_seq_len = past_cache.shape[1]
            seq_len = update.shape[1]
            
            for i in range(batch_size):
                start_idx = write_indices[i]
                for s in range(seq_len):
                    idx = start_idx + s
                    if mode == "circular":
                        idx = idx % max_seq_len
                    present_cache[i, idx, :] = update[i, s, :]
        elif past_cache.ndim == 4: 
            num_heads = past_cache.shape[1]
            max_seq_len = past_cache.shape[2]
            seq_len = update.shape[2]
            print(batch_size, num_heads, max_seq_len, seq_len)

            for i in range(batch_size):
                start_idx = write_indices[i]
                for h in range(num_heads):
                    for s in range(seq_len):
                        idx = start_idx + s
                        if mode == "circular":
                            idx = idx % max_seq_len
                        present_cache[i, h, idx, :] = update[i, h, s, :]
        else:
            raise ValueError(f"Unsupported past_cache dimension: {past_cache.ndim}")

        return (present_cache,)
