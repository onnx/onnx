# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,R0914,R0915,R1702,W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


def _conv_implementation(  # type: ignore
    X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides
):
    if dilations is None:
        dilations = [1 for s in X.shape[2:]]
    if kernel_shape is None:
        kernel_shape = W.shape[2:]
    if pads is None:
        pads = [0 for s in X.shape[2:]] * 2
    if strides is None:
        strides = [1 for s in X.shape[2:]]

    if X.shape[1] != W.shape[1] * group or W.shape[0] % group != 0:
        raise ValueError(
            f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}, "
            f"W should be {(W.shape[0], X.shape[1] / group, np.prod(W.shape[1:]) / X.shape[1] * group)}."
        )
    if group > 1:
        res = []
        td = 0
        mg = W.shape[0] // group
        dw = W.shape[1]
        for g in range(group):
            gx = X[:, g * dw : g * dw + dw]
            gw = W[g * mg : g * mg + mg]
            try:
                cv = _conv_implementation(
                    gx, gw, None, auto_pad, dilations, 1, kernel_shape, pads, strides
                )
            except ValueError as e:
                raise ValueError(
                    f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={g}/{group}, "
                ) from e
            td += cv.shape[1]
            res.append(cv)
        new_shape = list(res[0].shape)
        new_shape[1] = td
        final = np.zeros(tuple(new_shape), dtype=res[0].dtype)
        p = 0
        for g, cv in enumerate(res):
            final[:, p : p + cv.shape[1]] = cv
            p += cv.shape[1]
        if B is not None:
            new_shape = [1 for s in final.shape]
            new_shape[1] = B.shape[0]
            final += B.reshape(tuple(new_shape))
        return final

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

    if len(X.shape) == 4:
        sN, sC, sH, sW = X.shape
        # M, C_group, kH, kW = W.shape
        kh, kw = kernel_shape
        sth, stw = strides

        h_out = int(((sH - kh + pads[0] + pads[2]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[3]) / stw) + 1)

        h0, w0 = pads[0], pads[1]
        oh, ow = -1 * (kh % 2), -1 * (kw % 2)
        bh, bw = -h0, -w0
        eh, ew = h_out * sth, w_out * stw
        res = np.zeros((X.shape[0], W.shape[0]) + (h_out, w_out))
        if B is not None:
            res[:, :, :, :] = B.reshape((1, -1, 1, 1))  # type: ignore

        for n in range(0, sN):
            for nw in range(W.shape[0]):
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
                            w = W[nw : nw + 1, c : c + 1]
                            if img.shape != w.shape:
                                jh1, jh2 = max(-oh - i, 0), min(
                                    kh, kh + sH - (i + oh + kh)
                                )
                                jw1, jw2 = max(-ow - j, 0), min(
                                    kw, kw + sW - (j + ow + kw)
                                )
                                w = W[n : n + 1, c : c + 1, jh1:jh2, jw1:jw2]
                                if img.shape != w.shape:
                                    raise RuntimeError(
                                        f"Unexpected shape {img.shape} != {w.shape}, oh={oh}, ow={ow}, "
                                        f"i={i}, j={j}, kh={kh}, kw={kw}, sH={sH}, sW={sW}, sth={sth}, stw={stw}."
                                    )
                                s = (img * w).sum()
                            else:
                                s = (img * w).sum()
                            res[n, nw, hr, wr] += s  # type: ignore

        return res

    if len(X.shape) == 5:
        sN, sC, sH, sW, sZ = X.shape
        kh, kw, kz = kernel_shape
        sth, stw, stz = strides

        h_out = int(((sH - kh + pads[0] + pads[3]) / sth) + 1)
        w_out = int(((sW - kw + pads[1] + pads[4]) / stw) + 1)
        z_out = int(((sZ - kz + pads[2] + pads[5]) / stz) + 1)

        h0, w0, z0 = pads[0], pads[1], pads[2]
        oh, ow, oz = -1 * (kh % 2), -1 * (kw % 2), -1 * (kz % 2)
        bh, bw, bz = -h0, -w0, -z0
        eh, ew, ez = h_out * sth, w_out * stw, z_out * stz
        res = np.zeros((X.shape[0], W.shape[0]) + (h_out, w_out, z_out))
        if B is not None:
            res[:, :, :, :] = B.reshape((1, -1, 1, 1, 1))  # type: ignore

        for n in range(0, sN):
            for nw in range(W.shape[0]):
                for c in range(0, sC):
                    for io in range(bh, eh, sth):
                        for jo in range(bw, ew, stw):
                            for zo in range(bz, ez, stz):
                                hr, wr, zr = (
                                    (io - bh) // sth,
                                    (jo - bw) // stw,
                                    (zo - bz) // stz,
                                )
                                if hr >= h_out or wr >= w_out or zr >= z_out:
                                    continue
                                i = io + kh % 2
                                j = jo + kw % 2
                                z = zo + kz % 2
                                ih1, ih2 = max(0, i + oh), min(i + oh + kh, sH)
                                iw1, iw2 = max(0, j + ow), min(j + ow + kw, sW)
                                iz1, iz2 = max(0, z + oz), min(z + oz + kz, sZ)
                                img = X[n : n + 1, c : c + 1, ih1:ih2, iw1:iw2, iz1:iz2]
                                w = W[nw : nw + 1, c : c + 1]
                                if img.shape != w.shape:
                                    jh1, jh2 = max(-oh - i, 0), min(
                                        kh, kh + sH - (i + oh + kh)
                                    )
                                    jw1, jw2 = max(-ow - j, 0), min(
                                        kw, kw + sW - (j + ow + kw)
                                    )
                                    jz1, jz2 = max(-oz - j, 0), min(
                                        kz, kz + sZ - (z + oz + kz)
                                    )
                                    w = W[
                                        n : n + 1, c : c + 1, jh1:jh2, jw1:jw2, jz1:jz2
                                    ]
                                    if img.shape != w.shape:
                                        raise RuntimeError(
                                            f"Unexpected shape {img.shape} != {w.shape}, oh={oh}, ow={ow}, oz={oz}, "
                                            f"i={i}, j={j}, z={z}, kh={kh}, kw={kw}, kz={kz}, "
                                            f"sH={sH}, sW={sW}, sZ={sZ}, sth={sth}, stw={stw}, stz={stz}."
                                        )
                                    s = (img * w).sum()
                                else:
                                    s = (img * w).sum()
                                res[n, nw, hr, wr, zr] += s  # type: ignore

        return res

    if len(X.shape) == 3:
        sN, sC, sH = X.shape
        # M, C_group, kH, kW = W.shape
        (kh,) = kernel_shape
        (sth,) = strides

        h_out = int(((sH - kh + pads[0] + pads[1]) / sth) + 1)

        h0 = pads[0]
        oh = -1 * (kh % 2)
        bh = -h0
        eh = h_out * sth
        res = np.zeros((X.shape[0], W.shape[0]) + (h_out,))
        if B is not None:
            res[:, :, :] += B.reshape((1, -1, 1))  # type: ignore

        for n in range(0, sN):
            for nw in range(W.shape[0]):
                for c in range(0, sC):
                    for io in range(bh, eh, sth):
                        hr = (io - bh) // sth
                        if hr >= h_out:
                            continue
                        i = io + kh % 2
                        ih1, ih2 = max(0, i + oh), min(i + oh + kh, sH)
                        img = X[n : n + 1, c : c + 1, ih1:ih2]
                        w = W[nw : nw + 1, c : c + 1]
                        if img.shape != w.shape:
                            jh1, jh2 = max(-oh - i, 0), min(kh, kh + sH - (i + oh + kh))
                            w = w[:1, :1, jh1:jh2]
                            if img.shape != w.shape:
                                raise RuntimeError(
                                    f"Unexpected shape {img.shape} != {w.shape}, oh={oh}, "
                                    f"i={i}, kh={kh}, sH={sH}, sth={sth}."
                                )
                            s = (img * w).sum()
                        else:
                            s = (img * w).sum()
                        res[n, nw, hr] += s  # type: ignore

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
