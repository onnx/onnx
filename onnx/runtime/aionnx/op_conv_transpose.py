# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,R0914,R0915,R1702,W0221

import numpy as np  # type: ignore

from ..op_run import OpRun
from .op_col2im import col2im_naive_implementation


class ConvTranspose(OpRun):
    def _run(  # type: ignore
        self,
        X,
        W,
        B=None,
        auto_pad=None,
        dilations=None,
        group=None,
        kernel_shape=None,
        output_padding=None,
        output_shape=None,
        pads=None,
        strides=None,
    ):
        auto_pad = auto_pad or self.auto_pad  # type: ignore
        dilations = dilations or self.dilations  # type: ignore
        group = group or self.group  # type: ignore
        kernel_shape = kernel_shape or self.kernel_shape  # type: ignore
        output_padding = output_padding or self.output_padding  # type: ignore
        output_shape = output_shape or self.output_shape  # type: ignore
        pads = pads or self.pads  # type: ignore
        strides = strides or self.strides  # type: ignore

        if group != 1:
            raise RuntimeError(f"group={group} != 1 is not implemented yet.")
        if dilations is None:
            dilations = [1 for s in X.shape[2:]]
        if kernel_shape is None:
            kernel_shape = W.shape[2:]
        if pads is None:
            pads = [0 for s in X.shape[2:]] * 2
        if output_padding is None:
            output_padding = [0 for s in X.shape[2:]] * 2
        if strides is None:
            strides = [1 for s in X.shape[2:]]

        final = None
        for batch in range(X.shape[0]):

            gemm = np.matmul(X[batch], W)
            res = col2im_naive_implementation(
                gemm, output_shape, W.shape, dilations, pads, strides
            )
            if final is None:
                final = np.empty(X.shape[:1] + res.shape, dtype=X.dtype)
            final[batch, ...] = res[...]
        return (final,)
