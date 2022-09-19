# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class Conv(OpRun):
    def _run(
        self,
        X,
        W,
        B=None,
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

        if dilations is None:
            dilations = [1 for s in X.shape[2:]]
        if kernel_shape is None:
            kernel_shape = W.shape[2:]
        if pads is None:
            pads = [0 for s in X.shape[2:]]
        if strides is None:
            strides = [1 for s in X.shape[2:]]

        if min(dilations) != max(dilations):
            # Let's compute the dilated kernel.
            nd = len(dilations)
            new_kernel_shape = []
            new_shape = list(W.shape[:-nd])
            for i, d in enumerate(dilations):
                di = len(W.shape) - nd + i
                new_shape.append(W.shape[di] + (W.shape[di] - 1) * (d - 1))
                new_kernel_shape.append(
                    kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1)
                )
            new_w = np.zeros(tuple(new_shape), dtype=W.dtype)
            indices = [slice(0, new_w.shape[0]), slice(0, new_w.shape[1])]
            for i, d in enumerate(dilations):
                di = len(W.shape) - nd + i
                indices.append(slice(0, new_w.shape[di], d))
            new_w[tuple(indices)] = W
            W = new_w
            kernel_shape = new_kernel_shape

        if len(X.shape) == 4:
            sN, sC, sH, sW = X.shape
            # M, C_group, kH, kW = W.shape
            kh, kw = kernel_shape
            kh2, kw2 = kh // 2, kw // 2

            dh, dw = dilations
            h_out = int(((sH - kh + pads[0] + pads[2]) / strides[0]) + 1)
            w_out = int(((sW - kw + pads[1] + pads[3]) / strides[1]) + 1)
            h0, w0 = pads[0], pads[1]
            h1, w1 = pads[2], pads[3]
            res = np.zeros((X.shape[:2] + (h_out, w_out)))
            if B is not None:
                res[:, :, :, :] = B

            for n in range(sN):
                for c in range(sC):
                    for i in range(-h0 + 1, sH + h1 - 1):
                        if i + h0 - 1 >= res.shape[-2]:
                            continue
                        for j in range(-w0 + 1, sW + w1 - 1):
                            if j + w0 - 1 >= res.shape[-1]:
                                continue
                            img = X[
                                n : n + 1,
                                c : c + 1,
                                max(0, i - kh2) : min(i + kh2 + 1, sH),
                                max(0, j - kw2) : min(j + kw2 + 1, sW),
                            ]
                            if img.shape != W.shape:
                                w = W[
                                    n : n + 1,
                                    c : c + 1,
                                    max(kh2 - i, 0) : kh + min(0, sH - (i + kh2 + 1)),
                                    max(kw2 - j, 0) : kw + min(0, sW - (j + kw2 + 1)),
                                ]
                                s = (img * w).sum()
                            else:
                                s = (img * W).sum()
                            if B is not None:
                                s += B
                            res[n, c, i + h0 - 1, j + w0 - 1] = s

                return (res,)

        raise RuntimeError(
            f"The convolution for X.shape={X.shape}, W.shape={W.shape}, "
            f"kernel_shape={kernel_shape} is not implemented yet."
        )
