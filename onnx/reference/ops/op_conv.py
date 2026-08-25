# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _make_ind(dim, shape):
    m = np.empty(shape, dtype=np.int64)
    ind = [slice(0, shape[i]) for i in range(len(shape))]
    new_shape = [1] * len(shape)
    new_shape[dim] = shape[dim]
    first = np.arange(shape[dim]).reshape(new_shape)
    m[tuple(ind)] = first
    return m


def im2col(X, kernel_shape, pads, strides):
    n_dims = len(kernel_shape)
    m, n_C = X.shape[:2]

    kernel_size = np.prod(kernel_shape)
    shape_out = []
    for i, dim in enumerate(kernel_shape):
        dx = X.shape[2 + i]
        shape_out.append((dx + pads[i] + pads[i + n_dims] - dim) // strides[i] + 1)

    indices = []
    for i in range(len(shape_out)):
        kind = _make_ind(i, kernel_shape)
        iind = _make_ind(i, shape_out) * strides[i]
        index = np.tile(kind.ravel(), n_C).reshape(-1, 1) + iind.reshape(1, -1)
        indices.append(index)

    d = np.repeat(np.arange(n_C), kernel_size).reshape(-1, 1)

    nc = [(0, 0)] * 2
    padding = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
    X_padded = np.pad(X, tuple(nc) + tuple(padding), mode="constant")

    getitem = (slice(0, m), d, *indices)
    cols = X_padded[getitem]
    perm = (1, 0, *range(2, cols.ndim))
    output_size = m * int(np.prod(shape_out))
    return cols.transpose(perm).reshape((cols.shape[1], output_size)), tuple(shape_out)


def _conv_implementation(
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
    kernel_shape = tuple(kernel_shape)

    if X.shape[1] != W.shape[1] * group or W.shape[0] % group != 0:
        raise ValueError(
            f"Shape inconsistencies, X.shape={X.shape}, W.shape={W.shape}, group={group}, "
            f"W should be {(W.shape[0], X.shape[1] // group, np.prod(W.shape[1:]) // X.shape[1] * group)}."
        )
    if any(dilation != 1 for dilation in dilations):
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

    if auto_pad == "VALID":
        pads = [0] * (2 * len(kernel_shape))
    elif auto_pad in {"SAME_LOWER", "SAME_UPPER"}:
        head = []
        tail = []
        for i in range(len(X.shape) - 2):
            d = X.shape[i + 2]
            target_size = (d + strides[i] - 1) // strides[i]
            pad_needed = max(0, (target_size - 1) * strides[i] + kernel_shape[i] - d)
            if auto_pad == "SAME_LOWER":
                pad_head = (pad_needed + 1) // 2
            else:
                pad_head = pad_needed // 2
            pad_tail = pad_needed - pad_head
            head.append(pad_head)
            tail.append(pad_tail)
        pads = head + tail

    c2, out_shape = im2col(X, kernel_shape, pads, strides)
    kernel_size = int(np.prod(kernel_shape))
    flattened_kernel_size = W.shape[1] * kernel_size
    output_size = X.shape[0] * int(np.prod(out_shape))
    c2 = c2.reshape((group, flattened_kernel_size, output_size))
    w_reshaped = W.reshape((group, W.shape[0] // group, flattened_kernel_size))
    mul = w_reshaped @ c2
    mul = mul.reshape((group, W.shape[0] // group, X.shape[0], *out_shape))
    perm = (2, 0, 1, *range(3, mul.ndim))
    mul = mul.transpose(perm).reshape((X.shape[0], W.shape[0], *out_shape))

    if B is not None:
        if B.size == 1:
            return mul + B
        new_shape = [1] * len(mul.shape)
        new_shape[1] = -1
        mul += B.reshape(tuple(new_shape))
    return mul


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
        return (
            _conv_implementation(
                X, W, B, auto_pad, dilations, group, kernel_shape, pads, strides
            ).astype(X.dtype),
        )
