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
        if pads is not None:
            n_dims = len(pads) // 2
            new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
            if output_shape is None:
                output_shape = [
                    strides[i] * (X.shape[i + 2] - 1)
                    + output_padding[i]
                    + ((kernel_shape[i] - 1) * dilations[i] + 1)
                    - new_pads[i, 0]
                    - new_pads[i, 1]
                    for i in range(n_dims)
                ]
        else:
            total_padding = [
                strides[i] * (X.shape[i + 1] - 1)
                + output_padding[i]
                + ((kernel_shape[i] - 1) * dilations[i] + 1)
                - output_shape[i]
                for i in range(len(output_shape))
            ]
            pads = []
            for i in range(len(output_shape)):
                if auto_pad == "SAME_UPPER":
                    pads.append(total_padding[i] // 2)
                    pads.append(total_padding[i] - (total_padding[i] // 2))
                else:
                    pads.append(total_padding[i] - (total_padding[i] // 2))
                    pads.append(total_padding[i] // 2)

        kernel_shape = W.shape[2:]
        kernel_size = np.prod(kernel_shape)
        num_output_channels = W.shape[1] * group
        kernel_dim = num_output_channels // group * kernel_size

        C = X.shape[1]
        m = kernel_dim
        n = np.prod(X.shape[2:])
        k = C // group
        w_reshaped = W.reshape((group, m, k))
        final = None

        for image_id in range(X.shape[0]):
            for group_id in range(group):

                gemm = np.matmul(w_reshaped[group_id], X[0].reshape((k, n)))
                gemmc = gemm.reshape((num_output_channels, -1, gemm.shape[-1]))
                for c in range(num_output_channels):
                    res = col2im_naive_implementation(
                        gemmc[c], output_shape, kernel_shape, dilations, pads, strides
                    )
                    if final is None:
                        final = np.empty(
                            X.shape[:1] + (num_output_channels,) + res.shape,
                            dtype=X.dtype,
                        )
                    if B is not None:
                        res += B[c]
                    final[image_id, c, ...] = res[...]

        return (final,)
