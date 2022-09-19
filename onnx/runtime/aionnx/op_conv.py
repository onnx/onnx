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
        auto_pad = auto_pad or self.auto_pad  # type: ignore
        dilations = dilations or self.dilations  # type: ignore
        group = group or self.group  # type: ignore
        kernel_shape = kernel_shape or self.kernel_shape  # type: ignore
        pads = pads or self.pads  # type: ignore
        strides = strides or self.strides  # type: ignore

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
            sth, stw = strides

            dh, dw = dilations

            if auto_pad in {"SAME_LOWER", "SAME_UPPER", "VALID"}:
                head = []
                tail = []
                for i in range(len(X.shape) - 2):
                    d = X.shape[i]
                    target_size = (d + strides[i] - 1) // strides[i]
                    pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
                    if auto_pad == "SAME_LOWER":
                        pad_head = (pad_needed + 1) // 2
                    else:
                        pad_head = pad_needed // 2
                    pad_tail = pad_needed - pad_head
                    head.append(pad_head)
                    tail.append(pad_tail)
                pads = head + tail

            h_out = int(((sH - kh + pads[0] + pads[2]) / sth) + 1)
            w_out = int(((sW - kw + pads[1] + pads[3]) / stw) + 1)

            h0, w0 = pads[0], pads[1]
            h1, w1 = pads[2], pads[3]
            oh, ow = -1, -1
            res = np.zeros((X.shape[:2] + (h_out, w_out)))
            if B is not None:
                res[:, :, :, :] = B

            for n in range(0, sN):
                for c in range(0, sC):
                    for i in range(-h0 + 1, sH + h1 - 1, sth):
                        if i + h0 - 1 >= h_out * sth:
                            continue
                        for j in range(-w0 + 1, sW + w1 - 1, stw):
                            if j + w0 - 1 >= w_out * stw:
                                continue
                            img = X[
                                n : n + 1,
                                c : c + 1,
                                max(0, i + oh) : min(i + oh + kh, sH),
                                max(0, j + ow) : min(j + ow + kw, sW),
                            ]
                            if img.shape != W.shape:
                                w = W[
                                    n : n + 1,
                                    c : c + 1,
                                    max(-oh - i, 0) : min(kh, kh + sH - (i + oh + kh)),
                                    max(-ow - j, 0) : min(kw, kw + sW - (j + ow + kw)),
                                ]
                                if img.shape != w.shape:
                                    raise RuntimeError(
                                        f"Unexpected shape {img.shape} != {w.shape}, oh={oh}, ow={ow}, "
                                        f"i={i}, j={j}, kh={kh}, kw={kw}, sH={sH}, sW={sW}, sth={sth}, stw={stw}."
                                    )
                                s = (img * w).sum()
                            else:
                                s = (img * W).sum()
                            if B is not None:
                                s += B
                            res[n, c, (i + h0 - 1) // sth, (j + w0 - 1) // stw] = s

                return (res,)

        raise RuntimeError(
            f"The convolution for X.shape={X.shape}, W.shape={W.shape}, "
            f"kernel_shape={kernel_shape} is not implemented yet."
        )
