# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_conv import _conv_implementation


class CausalConvWithState_27(OpRun):
    def _run(self, input, weight, bias=None, past_state=None, activation=None):
        return self._compute(input, weight, bias, past_state, activation, 1)

    def _compute(
        self,
        input,
        weight,
        bias=None,
        past_state=None,
        activation=None,
        dilation=1,
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
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}.")
        if k < 1 or weight.shape[:2] != (channels, 1):
            raise ValueError(f"weight must have shape ({channels}, 1, k) with k >= 1.")
        history = (k - 1) * dilation
        if history > np.iinfo(np.int64).max:
            raise ValueError("History length exceeds int64.")
        if bias is not None and bias.shape != (channels,):
            raise ValueError(f"bias must have shape ({channels},).")
        if past_state is not None and past_state.shape != (
            batch_size,
            channels,
            history,
        ):
            raise ValueError(
                f"past_state must have shape ({batch_size}, {channels}, {history})."
            )

        # Retain raw chronological samples, including taps skipped by dilation.
        if past_state is None:
            pad = np.zeros((batch_size, channels, history), dtype=input.dtype)
        else:
            pad = past_state
        padded = np.concatenate([pad, input], axis=2)
        present_state = padded[:, :, padded.shape[2] - history :]
        if input.shape[2] == 0:
            return (input.copy(), present_state)

        # Step 2: depthwise Conv1d (group = channels, valid padding).
        conv_out = _conv_implementation(
            padded,
            weight,
            bias,
            "NOTSET",
            [dilation],
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

        return (output, present_state)


class CausalConvWithState_29(CausalConvWithState_27):
    def _run(
        self, input, weight, bias=None, past_state=None, activation=None, dilation=None
    ):
        return self._compute(
            input,
            weight,
            bias,
            past_state,
            activation,
            1 if dilation is None else dilation,
        )
