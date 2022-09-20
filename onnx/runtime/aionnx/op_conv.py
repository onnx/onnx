# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,R0914,R1702,W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


def _conv_implementation(  # type: ignore
    X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides
):
    if group != 1:
        raise RuntimeError(f"group={group} != 1 is not implemented yet.")
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]

    if dilations[0] != 1 or min(dilations) != max(dilations):
        # Let's compute the dilated kernel.
        nd = len(dilations)
        new_kernel_shape = []
        new_shape = list(W.shape[:-nd])
        for i, d in enumerate(dilations):
            di = len(W.shape) - nd + i
            new_shape.append(W.shape[di] + (W.shape[di] - 1) * (d - 1))
            new_kernel_shape.append(kernel_shape[i] + (kernel_shape[i] - 1) * (d - 1))
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
        oh, ow = -1 * (kh % 2), -1 * (kw % 2)
        bh, bw = -h0, -w0
        eh, ew = (h_out * sth) - h1, (w_out * stw) - w1
        res = np.zeros((X.shape[:2] + (h_out, w_out)))
        if B is not None:
            res[:, :, :, :] = B

        for n in range(0, sN):
            for c in range(0, sC):
                for io in range(bh, eh, sth):
                    for jo in range(bw, ew, stw):
                        hr, wr = (io - bh) // sth, (jo - bw) // stw
                        if hr >= h_out or wr >= w_out:
                            continue
                        i = io + kh % 2
                        j = jo + kw % 2
                        ih1, ih2 = max(0, i + oh), min(i + oh + kh, sH)
                        iw1, iw2 = max(0, j + ow), min(j + ow + kw, sW)
                        img = X[n : n + 1, c : c + 1, ih1:ih2, iw1:iw2]
                        if img.shape != W.shape:
                            jh1, jh2 = max(-oh - i, 0), min(kh, kh + sH - (i + oh + kh))
                            jw1, jw2 = max(-ow - j, 0), min(kw, kw + sW - (j + ow + kw))
                            w = W[n : n + 1, c : c + 1, jh1:jh2, jw1:jw2]
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
                        res[n, c, hr, wr] = s

            return res

    raise RuntimeError(
        f"The convolution for X.shape={X.shape}, W.shape={W.shape}, "
        f"kernel_shape={kernel_shape} is not implemented yet."
    )


class Conv(OpRun):
    def _run(  # type: ignore
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
        return (
            _conv_implementation(
                X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides
            ),
        )
