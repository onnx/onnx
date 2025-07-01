# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_conv import _conv_implementation


class ConvInteger(OpRun):
    def _run(
        self,
        X,
        W,
        x_zero_point=None,
        w_zero_point=None,
        auto_pad=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        pads=None,
        strides=None,
    ):
        if len(X.shape) < 3:
            raise ValueError(
                f"X must have at least 3 dimensions but its shape is {X.shape}."
            )
        auto_pad = auto_pad or self.auto_pad
        dilations = dilations or self.dilations
        group = group or self.group
        kernel_shape = kernel_shape or self.kernel_shape
        pads = pads or self.pads
        strides = strides or self.strides

        X = X.astype(np.int32)
        if x_zero_point:
            X -= x_zero_point
        W = W.astype(np.int32)
        if w_zero_point:
            W -= w_zero_point

        return (
            _conv_implementation(
                X, W, None, auto_pad, dilations, group, kernel_shape, pads, strides
            ).astype(np.int32),
        )
