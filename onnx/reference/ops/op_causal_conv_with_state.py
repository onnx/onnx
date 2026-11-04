# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_conv import _conv_implementation


class CausalConvWithState(OpRun):
    def _run(
        self,
        input,
        weight,
        bias=None,
        past_state=None,
        activation=None,
    ):
        if activation is None:
            activation = "none"
        if activation not in ("none", "silu", "swish"):
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                "Expected one of: 'none', 'silu', 'swish'."
            )

        if input.ndim != 3:
            raise ValueError(
                f"input must be rank 3 (batch_size, channels, length), got shape {input.shape}."
            )
        if weight.ndim != 3:
            raise ValueError(
                f"weight must be rank 3 (channels, 1, k), got shape {weight.shape}."
            )

        batch_size, channels, _ = input.shape
        k = weight.shape[2]

        # Step 1: build the left-padded input (B, C, L + k - 1).
        if past_state is None:
            pad = np.zeros((batch_size, channels, k - 1), dtype=input.dtype)
        else:
            pad = past_state
        padded = np.concatenate([pad, input], axis=2)

        # Step 2: depthwise Conv1d (group = channels, valid padding).
        conv_out = _conv_implementation(
            padded,
            weight,
            bias,
            "NOTSET",
            [1],
            channels,
            [k],
            [0, 0],
            [1],
        ).astype(input.dtype)

        # Step 3: optional fused SiLU/Swish activation.
        if activation in ("silu", "swish"):
            sigmoid = 1.0 / (1.0 + np.exp(-conv_out.astype(np.float32)))
            output = (conv_out.astype(np.float32) * sigmoid).astype(input.dtype)
        else:
            output = conv_out

        # Step 4: present_state = last (k - 1) positions of the padded input.
        present_state = padded[:, :, padded.shape[2] - (k - 1) :]

        return (output, present_state)
